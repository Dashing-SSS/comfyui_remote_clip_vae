// ===========================================================================
//  comfyui_remote_clip_vae / web/refresh.js
//
//  给三个「远程加载」节点加：
//     · 「🔄 刷新模型列表」整宽按钮 —— 一次刷新：服务端显卡 + 模型清单
//     · 「服务端显卡」只读 —— 值由节点自己从服务端读，手改会被还原，避免被误改坏
//  浏览器直接跨域请求会被挡，所以走 ComfyUI 本机代理 /remote_clipvae/probe（见 api.py）。
// ===========================================================================

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 每个节点：kind = 服务端 /health.models 里的分类；model = 下拉框的控件名；
// extra = 同一个节点上还要一起刷的下拉（提示词优化节点的「视觉模型」读 models.llm_vision）
const KIND_BY_NODE = {
  RemoteCLIPLoader: { kind: "clip", model: "模型" },
  RemoteVAELoader: { kind: "vae", model: "模型" },
  RemoteCLIPTextEncode: { kind: "clip", model: "模型" },
  RemoteCLIPImageEncode: { kind: "clip", model: "模型" },
  RemotePromptOptimizer: {
    kind: "llm",
    model: "文生图模型",
    share: ["图生图模型"],
    after: "API Key", // 刷新按钮插到这一栏后面
    extra: [{ kind: "llm_vision", model: "视觉模型", prefix: ["（不使用 / 纯文本）"] }],
  },
};

// 图片接口自动增槽的节点：接一张就在后面多出一个空接口（最多 IMG_MAX 张）
const IMG_NODES = new Set(["RemoteCLIPImageEncode", "RemotePromptOptimizer"]);

// 某个输入端一连上，就把节点自己的模型字段收起来（改用连过来的那个对象）
const LINK_HIDE = {
  RemoteCLIPTextEncode: {
    input: "clip",
    widgets: ["服务端显卡", "服务端地址", "模型", "手动模型名", "模型类型"],
  },
  RemoteCLIPImageEncode: {
    input: "clip",
    widgets: ["服务端显卡", "服务端地址", "模型", "手动模型名", "模型类型"],
  },
};

// 「🗑 卸载模型」按钮：靠上游节点（vae/clip）拿地址和模型名
const UNLOAD_EXTRA = {
  RemoteVAEDecode: { kind: "vae", input: "vae" },
  RemoteVAEEncode: { kind: "vae", input: "vae" },
};

// 本地/远程开关：两边都留 文生图模型 / 图生图模型 / 视觉模型 三栏，
// 只是候选换成「本机 models/LLM」或「服务端 models/LLM」，另加地址/Key/按钮的显隐
const MODE_TOGGLE = {
  RemotePromptOptimizer: {
    widget: "使用远程模型",
    remoteOnly: ["服务端地址", "API Key"],
    remoteButtons: ["🔄 刷新模型列表", "🗑 卸载模型"],
    combos: ["文生图模型", "图生图模型", "视觉模型"],
  },
};

async function localModelLists() {
  try {
    const resp = await api.fetchApi("/remote_clipvae/local_models", { cache: "no-store" });
    return await resp.json();
  } catch (e) {
    return { ok: false, t2i: [], i2i: [], llm_vision: [], error: String((e && e.message) || e) };
  }
}

function rememberMode(node, spec) {
  const remote = !!(getWidget(node, spec.widget) || {}).value;
  const vals = {};
  for (const nm of spec.combos || []) vals[nm] = (getWidget(node, nm) || {}).value;
  node.__rcvModeVals = node.__rcvModeVals || {};
  if (node.__rcvModeNow !== undefined && node.__rcvModeNow !== remote) {
    node.__rcvModeVals[node.__rcvModeNow ? "remote" : "local"] = vals;
  }
  node.__rcvModeNow = remote;
  return remote;
}

async function applyModeToggle(node, spec, { refill = true } = {}) {
  const modeW = getWidget(node, spec.widget);
  if (!modeW) return;
  const remote = rememberMode(node, spec);
  for (const nm of spec.remoteOnly) setWidgetHidden(node, nm, !remote);
  for (const nm of spec.remoteButtons || []) setWidgetHidden(node, nm, !remote);
  try {
    const s = node.computeSize();
    node.setSize([Math.max(node.size[0], s[0]), Math.max(160, s[1])]);
    node.setDirtyCanvas(true, true);
  } catch (e) {}
  if (!refill) return;

  const saved = (node.__rcvModeVals || {})[remote ? "remote" : "local"] || {};
  if (!remote) {
    // 本地模式：三个下拉换成扫到的本机 GGUF
    const d = await localModelLists();
    const t2i = (d && d.ok ? d.t2i || [] : []).slice();
    const i2i = (d && d.ok ? d.i2i || [] : []).slice();
    const vis = (d && d.ok ? d.llm_vision || [] : []).slice();
    if (t2i.length) {
      setComboValues(getWidget(node, "文生图模型"), t2i, saved["文生图模型"]);
      setComboValues(getWidget(node, "图生图模型"), i2i.length ? i2i : t2i, saved["图生图模型"]);
      // 本机只有一个 mmproj 时直接选上（接图就能跑，不用再手动点一次）
      let visKeep = saved["视觉模型"];
      if (vis.length === 1 && (!visKeep || String(visKeep).startsWith("（"))) visKeep = vis[0];
      setComboValues(getWidget(node, "视觉模型"), ["（不使用 / 纯文本）"].concat(vis), visKeep);
    } else {
      const hint = d && d.error ? d.error : "（本机 models/LLM 里没扫到 GGUF）";
      setComboValues(getWidget(node, "文生图模型"), [hint]);
      setComboValues(getWidget(node, "图生图模型"), [hint]);
      setComboValues(getWidget(node, "视觉模型"), ["（不使用 / 纯文本）"]);
    }
    if (vis.length || !t2i.length) {
      node.setDirtyCanvas(true, true);
    }
    return;
  }
  // 远程模式：把三个下拉从服务端拉回来
  const kind = KIND_BY_NODE[node.name] || KIND_BY_NODE.RemotePromptOptimizer;
  await probe(node, kind, { silent: true });
  for (const nm of spec.combos || []) {
    const w = getWidget(node, nm);
    if (w && saved[nm] !== undefined && (w.options?.values || []).includes(saved[nm])) w.value = saved[nm];
  }
  node.setDirtyCanvas(true, true);
}
const IMG_RE = /^图片(\d+)$/;
const IMG_MAX = 10;

function imgInputsOf(node) {
  const list = [];
  for (const inp of node.inputs || []) {
    const m = IMG_RE.exec(inp.name || "");
    if (m) list.push({ n: Number(m[1]), input: inp });
  }
  return list.sort((a, b) => a.n - b.n);
}

// 只保留「已连接的最大序号 + 1」个接口，少一个就补
function syncImgSlots(node) {
  if (!node || node.__rcvSyncing) return;
  node.__rcvSyncing = true;
  try {
    const items = imgInputsOf(node);
    let maxLink = 0;
    for (const it of items) if (it.input.link != null) maxLink = Math.max(maxLink, it.n);
    const want = Math.min(IMG_MAX, Math.max(1, maxLink + 1));

    for (let i = (node.inputs || []).length - 1; i >= 0; i--) {
      const m = IMG_RE.exec(node.inputs[i].name || "");
      if (m && Number(m[1]) > want) node.removeInput(i);
    }

    const have = new Set(imgInputsOf(node).map((it) => it.n));
    let lastIndex = (node.inputs || []).length;
    for (let n = 1; n <= want; n++) {
      if (have.has(n)) continue;
      node.addInput(`图片${n}`, "IMAGE");
      const slot = node.inputs.length - 1;
      if (slot >= lastIndex) lastIndex = slot;
    }
    grow(node);
  } finally {
    node.__rcvSyncing = false;
  }
}

const GPU_W = "服务端显卡";
const URL_W = "服务端地址";
const IF_W = "接口类型";
const TOKEN_W = "token";
const MODEL_W = "模型";
const BTN = "🔄 刷新模型列表";

const FAIL_TEXT = "（连接失败，检查服务端）";
const EMPTY_TEXT = "（没有可用模型，去服务端加目录后重扫）";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function grow(node) {
  try {
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
    node.setDirtyCanvas(true, true);
  } catch (e) {}
}

// 收起 / 展开某个控件：收起后不占高度、也点不到（值仍随工作流保存）
function setWidgetHidden(node, name, hidden) {
  const w = getWidget(node, name);
  if (!w) return;
  if (!w.__rcvOrig) w.__rcvOrig = { type: w.type, computeSize: w.computeSize };
  if (hidden) {
    w.type = "hidden";
    w.computeSize = () => [0, -4];
  } else {
    w.type = w.__rcvOrig.type;
    if (w.__rcvOrig.computeSize) w.computeSize = w.__rcvOrig.computeSize;
    else delete w.computeSize;
  }
}

function inputLinked(node, name) {
  const inp = (node.inputs || []).find((i) => i.name === name);
  return !!(inp && inp.link != null);
}

// 顺着连线找到上游节点上的某个控件值（VAE 解码/编码没有地址栏，就靠它拿）
function upstreamWidget(node, inputName, widgetName) {
  const inp = (node.inputs || []).find((i) => i.name === inputName);
  if (!inp || inp.link == null) return "";
  const graph = node.graph || app.graph;
  if (!graph) return "";
  const link = (graph.links || {})[inp.link];
  if (!link) return "";
  const src =
    (graph.getNodeById && graph.getNodeById(link.origin_id)) ||
    ((graph._nodes_by_id || {})[link.origin_id]) ||
    null;
  if (!src) return "";
  const w = getWidget(src, widgetName);
  return w ? w.value : "";
}

const UNLOAD_BTN = "🗑 卸载模型";

function unloadTarget(node, spec) {
  const addrW = getWidget(node, URL_W);
  const modelW = spec.model ? getWidget(node, spec.model) : getWidget(node, MODEL_W);
  const input = spec.input || "vae";
  const addr = ((addrW && addrW.value) || upstreamWidget(node, input, URL_W) || "").trim();
  let model = ((modelW && modelW.value) || upstreamWidget(node, input, MODEL_W) || "").trim();
  if (spec.kind === "llm" || String(model).startsWith("（") || String(model).startsWith("(")) model = "";
  return { addr, model };
}

async function doUnload(node, spec) {
  const { addr, model } = unloadTarget(node, spec);
  if (!addr) {
    alert("先填「服务端地址」，或者在左边接一个远程加载器（地址会从上游节点取）");
    return;
  }
  let data = { ok: false, error: "无响应" };
  try {
    const resp = await api.fetchApi(
      `/remote_clipvae/unload?url=${encodeURIComponent(addr)}&token=` +
        `&kind=${encodeURIComponent(spec.kind || "clip")}&model=${encodeURIComponent(model)}`,
      { cache: "no-store" }
    );
    data = await resp.json();
  } catch (e) {
    data = { ok: false, error: String((e && e.message) || e) };
  }
  const what = spec.kind === "llm" ? "GGUF 模型" : `${(spec.kind || "").toUpperCase()} 模型`;
  if (data && data.ok) {
    const freed = (data.freed || []).join(", ");
    alert(`已让服务端卸载${what}${model ? "：" + model : ""}\n释放：${freed || "（服务端本来就没加载）"}`);
  } else {
    alert("卸载失败：\n" + ((data && data.error) || ""));
  }
}

function applyLinkHide(node, spec) {
  const linked = inputLinked(node, spec.input);
  for (const nm of spec.widgets) setWidgetHidden(node, nm, linked);
  try {
    const s = node.computeSize();
    node.setSize([Math.max(node.size[0], s[0]), Math.max(160, s[1])]);
    node.setDirtyCanvas(true, true);
  } catch (e) {}
}

function setComboValues(widget, values, keep) {
  if (!widget) return;
  if (!widget.options) widget.options = {};
  widget.options.values = values;
  const cur = keep !== undefined ? keep : widget.value;
  widget.value = values.includes(cur) ? cur : values.length ? values[0] : "";
}

function setText(widget, text) {
  if (!widget || widget.value === text) return;
  widget.value = text;
  if (widget.inputEl) widget.inputEl.value = text;
}

// 「服务端显卡」只读：手改立刻被还原
function lockGpuWidget(node) {
  const w = getWidget(node, GPU_W);
  if (!w) return;
  const auto = { value: w.value || "" };
  const origCb = w.callback;
  w.callback = function () {
    const out = origCb ? origCb.apply(this, arguments) : undefined;
    if (this.value !== auto.value) {
      this.value = auto.value;
      node.setDirtyCanvas(true, true);
    }
    return out;
  };
  if (w.inputEl) {
    w.inputEl.readOnly = true;
    w.inputEl.style.opacity = "0.7";
  }
  node.__rcvSetGpu = (text) => {
    auto.value = text;
    setText(w, text);
    node.setDirtyCanvas(true, true);
  };
}

function setGpu(node, text) {
  if (typeof node.__rcvSetGpu === "function") node.__rcvSetGpu(text);
  else setText(getWidget(node, GPU_W), text);
}

function query(node, kind) {
  const urlW = getWidget(node, URL_W);
  const tokenW = getWidget(node, TOKEN_W);
  return {
    url: ((urlW && urlW.value) || "").trim(),
    token: (tokenW && tokenW.value) || "",
    kind,
  };
}

async function fetchKind(q, kind) {
  try {
    const resp = await api.fetchApi(
      `/remote_clipvae/probe?url=${encodeURIComponent(q.url)}` +
        `&token=${encodeURIComponent(q.token)}&kind=${kind}`,
      { cache: "no-store" }
    );
    return await resp.json();
  } catch (e) {
    return { ok: false, error: String((e && e.message) || e) };
  }
}

async function probe(node, spec, { silent = true } = {}) {
  const mainName = spec.model || MODEL_W;
  const targets = [mainName].concat(spec.share || []);
  const q = query(node, spec.kind);
  if (!q.url) {
    setGpu(node, "（先填服务端地址）");
    return;
  }

  const data = await fetchKind(q, spec.kind);
  const models = (data && data.ok ? data.models || [] : []).slice();

  if (data && data.ok) {
    setGpu(node, data.device || "（服务端没报显卡型号）");
    if (models.length) {
      for (const nm of targets) {
        const w = getWidget(node, nm);
        if (w) setComboValues(w, models, w.value);
      }
      if (!silent) console.log(`[远程CLIP/VAE] 已刷新 ${models.length} 个 ${spec.kind.toUpperCase()} 模型：`, models);
    } else {
      for (const nm of targets) setComboValues(getWidget(node, nm), [EMPTY_TEXT]);
      if (!silent) alert("服务端没有可用模型：\n" + (data.error || ""));
    }
  } else {
    setGpu(node, "（连不上服务端）");
    for (const nm of targets) setComboValues(getWidget(node, nm), [FAIL_TEXT]);
    if (!silent) alert("刷新失败：\n" + ((data && data.error) || ""));
  }

  // 同一节点上的附加下拉（比如提示词优化的「视觉模型」）
  for (const ex of spec.extra || []) {
    const w = getWidget(node, ex.model);
    if (!w) continue;
    const d2 = await fetchKind(q, ex.kind);
    const names = (d2 && d2.ok ? (d2.models || []) : []).slice();
    const keep = w.value;
    setComboValues(w, (ex.prefix || []).concat(names), keep);
    if (!names.length && !silent) console.warn(`[远程CLIP/VAE] ${ex.kind} 为空：`, (d2 && d2.error) || "");
  }
  grow(node);
}

app.registerExtension({
  name: "RemoteCLIPVAE.RefreshModels",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const kind = KIND_BY_NODE[nodeData.name];
    const unloadSpec = kind || UNLOAD_EXTRA[nodeData.name];
    const hasImgSlots = IMG_NODES.has(nodeData.name);
    const hideSpec = LINK_HIDE[nodeData.name];
    const modeSpec = MODE_TOGGLE[nodeData.name];
    if (!kind && !hasImgSlots && !hideSpec && !unloadSpec && !modeSpec) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const node = this;

      if (kind) {
        // 1) 显卡栏只读
        lockGpuWidget(node);

        // 2) 整宽刷新按钮：显卡 + 模型一起刷
        const btn = node.addWidget("button", BTN, null, () => probe(node, kind, { silent: false }));
        if (btn) {
          btn.serialize = false; // 按钮状态不写进工作流
          btn.options = btn.options || {};
          // 提示词优化要把刷新按钮插在「API Key」正下方，和参考节点的排布一致
          if (kind.after) {
            const cur = node.widgets.indexOf(btn);
            if (cur >= 0) node.widgets.splice(cur, 1);
            const at = node.widgets.findIndex((w) => w.name === kind.after);
            node.widgets.splice(at >= 0 ? at + 1 : node.widgets.length, 0, btn);
          }
        }
        // 3) 改地址后自动重探（防抖 900ms，静默不打扰）
        const urlW = getWidget(node, URL_W);
        if (urlW) {
          const origCb = urlW.callback;
          let timer = null;
          urlW.callback = function () {
            const out = origCb ? origCb.apply(this, arguments) : undefined;
            if (timer) clearTimeout(timer);
            timer = setTimeout(() => probe(node, kind, { silent: true }), 900);
            return out;
          };
        }

        // 4) 首次创建：先读显卡；下拉还是占位符时静默拉一次模型
        setTimeout(() => {
          const q = query(node, kind);
          if (!q.url) return;
          probe(node, kind, { silent: true });
        }, 600);
      }

      // 5) 图片接口自动增槽（图文编码 / 提示词优化）
      if (hasImgSlots) {
        // 等这一轮图加载/连线恢复后再收口（新节点没有连线 → 只留 图片1 一个空槽）
        setTimeout(() => syncImgSlots(node), 80);
      }

      // 6) 「clip」输入端连了东西 → 收起本节点自己的模型字段
      if (hideSpec) setTimeout(() => applyLinkHide(node, hideSpec), 60);

      // 7) 「🗑 卸载模型」按钮：紧跟在「卸载模型」下拉后面
      if (unloadSpec) {
        const ub = node.addWidget("button", UNLOAD_BTN, null, () => doUnload(node, unloadSpec));
        if (ub) {
          ub.serialize = false;
          ub.options = ub.options || {};
          const cur = node.widgets.indexOf(ub);
          if (cur >= 0) node.widgets.splice(cur, 1);
          const at = node.widgets.findIndex((w) => w.name === "卸载模型");
          node.widgets.splice(at >= 0 ? at + 1 : node.widgets.length, 0, ub);
        }
      }

      // 8) 本地/远程开关：立刻收起另一半
      if (modeSpec) {
        const modeW = getWidget(node, modeSpec.widget);
        if (modeW) {
          const orig = modeW.callback;
          modeW.callback = function () {
            const out = orig ? orig.apply(this, arguments) : undefined;
            setTimeout(() => applyModeToggle(node, modeSpec), 0);
            return out;
          };
        }
        setTimeout(() => applyModeToggle(node, modeSpec), 150);
      }

      grow(node);
      return r;
    };

    if (hasImgSlots || hideSpec) {
      const onCC = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function () {
        const r0 = onCC ? onCC.apply(this, arguments) : undefined;
        if (hasImgSlots && !this.__rcvSyncing) setTimeout(() => syncImgSlots(this), 0);
        if (hideSpec) setTimeout(() => applyLinkHide(this, hideSpec), 0);
        return r0;
      };
    }

    // 载入已有工作流时也补一次（节点上的值有可能是旧的）
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      if (kind) setTimeout(() => probe(this, kind, { silent: true }), 900);
      if (hasImgSlots) setTimeout(() => syncImgSlots(this), 120);
      if (hideSpec) setTimeout(() => applyLinkHide(this, hideSpec), 120);
      if (modeSpec) setTimeout(() => applyModeToggle(this, modeSpec), 350);
      return r;
    };
  },
});
