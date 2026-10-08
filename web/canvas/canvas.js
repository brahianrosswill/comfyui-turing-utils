import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREFIX = "TuringCanvas";
const isCard = n => (n?.comfyClass ?? n?.type ?? "").startsWith(PREFIX);
const manager = () => app.graph?._nodes?.find(n => n.type === "TuringCanvasSettings");
const value = (n, key) => n.widgets?.find(w => w.name === key)?.value;
let armed = false;
let busy = false;
let refreshTimer;
let lastState = {tasks: {}};
let lastPlan = [];
let configuring = false;
let filteredState;
let activeTask = null;

function snapshot() {
  return {nodes: app.graph._nodes.map(n => ({
    id: String(n.id), type: n.comfyClass ?? n.type, mode: n.mode,
    values: Object.fromEntries((n.widgets ?? []).filter(w => typeof w.value !== "function" && w.type !== "button" && !w.canvasPreview)
      .map(w => [w.name, w.value])),
    inputs: Object.fromEntries((n.inputs ?? []).filter(i => i.link != null).map(i => {
      const link = app.graph.links[i.link];
      return [i.name, {node: String(link.origin_id), slot: link.origin_slot,
        ...(n.properties?.canvasPinned?.[i.name] ? {asset: n.properties.canvasPinned[i.name]} : {})}];
    })),
  }))};
}

async function request(path, body) {
  const response = await api.fetchApi(`/turing/canvas/${path}`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : JSON.stringify(data));
  return data;
}

function report(error) { console.error("Turing Canvas", error); window.alert(error.message ?? String(error)); }

function preview(node, asset) {
  if (!node.canvasMedia || !asset || !manager()) return;
  const directory = value(manager(), "work_directory");
  const url = api.apiURL(`/turing/canvas/asset?directory=${encodeURIComponent(directory)}&id=${encodeURIComponent(asset)}${node.type === 'TuringCanvasMask' ? '&preview=1' : ''}`);
  if (node.canvasMedia.dataset.assetUrl === url) return;
  node.canvasMedia.dataset.assetUrl = url;
  node.canvasMedia.src = url;
}

async function refresh() {
  if (!manager()) { armed = false; return; }
  const data = await request("state", {graph: snapshot()});
  lastState = data.state;
  lastPlan = data.plan;
  for (const node of app.graph._nodes.filter(isCard)) {
    const result = lastState.tasks[String(node.id)];
    if (result) {
      node.properties.canvasAsset = result.asset;
      preview(node, result.asset);
    } else if (["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(node.type)) {
      preview(node, value(node, "asset_id"));
    }
    node.canvasStatus = lastPlan.find(p => p.id === String(node.id))?.status ?? "material";
  }
  app.graph.setDirtyCanvas(true, true);
}

async function waitResult(id) {
  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const response = await api.fetchApi(`/history/${id}`);
    const history = (await response.json())[id];
    if (history) {
      if (history.status?.status_str !== "success") throw new Error(`Task ${id} failed or was interrupted; previous result preserved.`);
      return;
    }
    const queue = await (await api.fetchApi("/queue")).json();
    if (![...queue.queue_running, ...queue.queue_pending].some(q => q[1] === id)) {
      // Check history again to cover the queue->history transition.
      const done = (await (await api.fetchApi(`/history/${id}`)).json())[id];
      if (done?.status?.status_str === "success") return;
      throw new Error("Task failed, was cancelled, or removed from the queue; previous result preserved.");
    }
  }
}

async function runTask(node, graph = snapshot()) {
  node.canvasRunning = true;
  app.graph.setDirtyCanvas(true, true);
  try {
    const result = await request("run", {graph, task: String(node.id), client_id: api.clientId});
    activeTask = {node, promptId: result.prompt_id};
    await waitResult(result.prompt_id);
    node.canvasError = false;
    await refresh();
  } catch (error) { node.canvasError = true; throw error; }
  finally { activeTask = null; node.canvasRunning = false; node.canvasProgress = ""; app.graph.setDirtyCanvas(true, true); }
}

async function generate(node) {
  if (busy) throw new Error("A canvas task is already running. Wait for it to finish.");
  busy = true;
  try { await runTask(node); } finally { busy = false; }
}

async function globalRefresh() {
  if (!armed) { app.extensionManager?.toast?.add({severity: "info", summary: "Canvas refresh is locked", detail: "Prepare refresh in Canvas Settings first.", life: 3500}); return; }
  if (busy) throw new Error("A canvas task is already running");
  const graph = snapshot();
  const data = await request("state", {graph});
  const jobs = data.plan.filter(p => ["changed", "upstream"].includes(p.status));
  if (data.plan.some(p => p.status === "blocked")) throw new Error("Canvas has missing inputs. Resolve blocked cards before global refresh.");
  if (!window.confirm(`Generate ${jobs.length} material tasks once? Model/prompt-only changes are not included.`)) return;
  armed = false;
  busy = true;
  try {
    for (const job of jobs) await runTask(app.graph.getNodeById(Number(job.id)) ?? app.graph.getNodeById(job.id), graph);
  } finally { busy = false; armed = false; }
}

async function importFile(node, file) {
  const settings = manager();
  if (!settings) throw new Error("Create Canvas Settings first");
  if (value(settings, "import_mode") !== "browser_upload") {
    throw new Error("Local-copy mode: browsers cannot expose the server path of a dropped file. Put it in the server input directory, enter local_path, then click Load / Refresh. No upload was started.");
  }
  const form = new FormData();
  form.append("canvas", JSON.stringify({graph: snapshot(), task: String(node.id)}));
  form.append("file", file);
  const response = await api.fetchApi("/turing/canvas/upload", {method: "POST", body: form});
  const asset = await response.json();
  if (!response.ok) throw new Error(asset.error);
  node.widgets.find(w => w.name === "asset_id").value = asset.id;
  await refresh();
}

function button(node, name, callback) {
  node.addWidget("button", name, null, () => Promise.resolve().then(callback).catch(report), {serialize: false});
}

function filterTypes() {
  const active = !!manager();
  for (const [type, constructor] of Object.entries(LiteGraph.registered_node_types)) {
    if (!Object.hasOwn(constructor, "canvasOriginalSkip")) constructor.canvasOriginalSkip = constructor.skip_list;
    constructor.skip_list = active ? !type.startsWith(PREFIX) : constructor.canvasOriginalSkip;
  }
  // Frontend 1.53 exposes this store through Pinia, but not comfyAPI.
  // Optional bridge: creation/backend validation remain enforced if it moves.
  const store = window.comfyAPI?.nodeDefStore?.useNodeDefStore?.()
    ?? app.extensionManager?._p?._s?.get("nodeDef");
  if (store?.registerNodeDefFilter && filteredState !== active) {
    store.unregisterNodeDefFilter("turing.canvas");
    if (active) store.registerNodeDefFilter({id: "turing.canvas", predicate: def => def.name.startsWith(PREFIX)});
    filteredState = active;
  }
}

app.registerExtension({
  name: "TuringUtils.MaterialCanvas",
  async setup() {
    api.addEventListener("progress", event => {
      if (!activeTask || (event.detail.prompt_id && event.detail.prompt_id !== activeTask.promptId)) return;
      activeTask.node.canvasProgress = `${event.detail.value}/${event.detail.max}`;
      app.graph.setDirtyCanvas(true, true);
    });
    const queue = app.queuePrompt;
    app.queuePrompt = function (...args) {
      if (!manager() && !app.graph._nodes.some(isCard)) return queue.apply(this, args);
      return globalRefresh().catch(report);
    };
    const create = LiteGraph.createNode;
    LiteGraph.createNode = function(type, ...args) {
      if (manager() && !type.startsWith(PREFIX) && !configuring) {
        app.extensionManager?.toast?.add({severity: "warn", summary: "Canvas-only workspace", detail: "Ordinary nodes cannot be created here.", life: 3500});
        return null;
      }
      return create.call(this, type, ...args);
    };
    setInterval(filterTypes, 1000);
    setInterval(() => { if (manager() && !busy) refresh().catch(() => {}); }, 3000);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name.startsWith("_TuringCanvas")) nodeType.skip_list = true;
  },
  nodeCreated(node) {
    if (!isCard(node)) return;
    const type = node.comfyClass ?? node.type;
    node.properties ??= {};
    node.size = [390, 350];
    const draw = node.onDrawForeground;
    node.onDrawForeground = function(ctx, ...args) {
      draw?.call(this, ctx, ...args);
      const status = this.canvasRunning ? "running" : this.canvasError ? "failed" : this.canvasStatus;
      const colors = {running: "#53a7ff", failed: "#ee6565", blocked: "#ee6565", changed: "#e3b64e", upstream: "#e3b64e"};
      if (colors[status]) {
        ctx.save(); ctx.strokeStyle = colors[status]; ctx.lineWidth = 3;
        if (status === "upstream") ctx.setLineDash([8, 5]);
        ctx.strokeRect(0, 0, this.size[0], this.size[1]); ctx.restore();
      }
      ctx.save(); ctx.fillStyle = "#ccc"; ctx.font = "12px sans-serif";
      ctx.fillText(`${status ?? "material"} ${this.canvasProgress ?? ""}${armed ? " · refresh armed" : ""}`, 8, -8); ctx.restore();
    };
    if (type === "TuringCanvasSettings") {
      const name = `canvas/${crypto.randomUUID()}`;
      for (const key of ["work_directory", "cache_directory"]) {
        const widget = node.widgets.find(w => w.name === key);
        if (widget) widget.value = name;
      }
      button(node, "Prepare / Lock Global Refresh", async () => {
        if (busy) throw new Error("Wait for the current canvas run before preparing another refresh");
        await refresh(); armed = !armed; app.graph.setDirtyCanvas(true, true);
      });
      button(node, "Refresh Changed Materials Once", globalRefresh);
      button(node, "Save Canvas Project", async () => request("save", {graph: snapshot(), workflow: app.graph.serialize()}));
      button(node, "Open Saved Project", async () => {
        if (!window.confirm("Replace the current canvas with the saved project?")) return;
        await app.loadGraphData(await request("load", {graph: snapshot()}));
      });
    } else if (["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(type)) {
      button(node, "Load / Refresh", async () => {
        if (value(manager(), "import_mode") === "local_copy") {
          const asset = await request("import", {graph: snapshot(), task: String(node.id)});
          node.widgets.find(w => w.name === "asset_id").value = asset.id;
          await refresh();
        } else {
          const input = document.createElement("input"); input.type = "file";
          input.accept = type === "TuringCanvasImage" ? "image/*" : type === "TuringCanvasVideo" ? "video/*" : "audio/*";
          input.onchange = () => input.files[0] && importFile(node, input.files[0]).catch(report);
          input.click();
        }
      });
      node.onDragOver = () => true;
      node.onDragDrop = function(event) {
        if (!event.dataTransfer?.files?.length) return false;
        importFile(node, event.dataTransfer.files[0]).catch(report);
        return true;
      };
    } else {
      button(node, "Generate New Result", () => generate(node));
      if (type === "TuringCanvasH3") button(node, "Enhance User Prompt", async () => {
        const widget = node.widgets.find(w => w.name === "model_prompt");
        const before = widget.value;
        const result = await request("enhance", {graph: snapshot(), task: String(node.id)});
        if (widget.value !== before && !window.confirm("Model prompt was edited during enhancement. Replace it?")) return;
        node.properties.canvasPreviousPrompt = widget.value;
        widget.value = result.model_prompt;
        app.graph.setDirtyCanvas(true, true);
      });
      button(node, "Select Published Version", async () => {
        await refresh();
        const history = lastState.tasks[String(node.id)]?.history ?? [];
        const answer = window.prompt(history.map((h, i) => `${i + 1}: ${h.asset}`).join("\n"), String(history.length));
        if (answer === null) return;
        const item = history[Number(answer) - 1];
        if (!item) throw new Error("Select a valid version number");
        await request("select", {graph: snapshot(), task: String(node.id), asset: item.asset});
        await refresh();
      });
    }
    if (type !== "TuringCanvasSettings") {
      const tag = ["TuringCanvasImage", "TuringCanvasMask"].includes(type) ? "img" : type === "TuringCanvasAudio" ? "audio" : "video";
      const media = document.createElement(tag);
      media.style.cssText = "width:100%;height:100%;object-fit:contain;background:#161616";
      if (tag !== "img") { media.controls = true; media.preload = "metadata"; }
      media.addEventListener("dragover", event => event.preventDefault());
      media.addEventListener("drop", event => {
        if (!["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(type)) return;
        event.preventDefault(); event.stopPropagation();
        if (event.dataTransfer.files[0]) importFile(node, event.dataTransfer.files[0]).catch(report);
      });
      const widget = node.addDOMWidget("material_preview", "canvas_preview", media, {serialize: false, hideOnZoom: false});
      widget.canvasPreview = true;
      widget.computeSize = () => [350, tag === "audio" ? 65 : 220];
      node.canvasMedia = media;
    }
    const oldMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(_, options) {
      oldMenu?.apply(this, arguments);
      if (this.properties.canvasPreviousPrompt !== undefined) options.push({content: "Undo prompt enhancement", callback: () => {
        this.widgets.find(w => w.name === "model_prompt").value = this.properties.canvasPreviousPrompt;
      }});
      for (const input of this.inputs ?? []) if (input.link != null) {
        options.push({content: `Pin / Follow: ${input.name}`, callback: async () => {
          this.properties.canvasPinned ??= {};
          if (this.properties.canvasPinned[input.name]) delete this.properties.canvasPinned[input.name];
          else {
            const source = app.graph.getNodeById(app.graph.links[input.link].origin_id);
            const asset = value(source, "asset_id") || source.properties.canvasAsset;
            if (!asset) return report(new Error("Source has no published material"));
            this.properties.canvasPinned[input.name] = asset;
          }
          await refresh();
        }});
      }
    };
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => { filterTypes(); refresh().catch(() => {}); }, 500);
  },
  beforeConfigureGraph() { configuring = true; },
  afterConfigureGraph() { configuring = false; armed = false; filterTypes(); refresh().catch(() => {}); },
});
