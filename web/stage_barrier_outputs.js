import { app } from "../../scripts/app.js";

const NODE_TYPE = "TuringUtilsStageBarrier";
const VALUE_INPUT = /^values\.value_(\d+)$/;
const ANY_TYPE = "*";
const installed = new WeakSet();

function valueInputs(node) {
  return node.inputs
    .map((input) => {
      const match = VALUE_INPUT.exec(input.name);
      return match ? { input, index: Number(match[1]) } : null;
    })
    .filter(Boolean)
    .sort((left, right) => left.index - right.index);
}

function syncOutputs(node) {
  const inputs = valueInputs(node);
  const outputCount = inputs.length ? inputs.at(-1).index + 1 : 0;

  while (node.outputs.length > outputCount) {
    node.removeOutput(node.outputs.length - 1);
  }
  while (node.outputs.length < outputCount) {
    node.addOutput(`value_${node.outputs.length}`, ANY_TYPE);
  }

  node.outputs.forEach((output, index) => {
    const input = inputs.find((entry) => entry.index === index)?.input;
    output.name = `value_${index}`;
    output.label = input?.label || input?.name?.split(".").at(-1) || output.name;
    output.type = input?.type || ANY_TYPE;
  });

  const size = node.computeSize();
  node.setSize([Math.max(node.size?.[0] ?? size[0], size[0]), size[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

function scheduleSync(node) {
  requestAnimationFrame(() => syncOutputs(node));
}

function install(node) {
  if (installed.has(node)) return;
  installed.add(node);

  const onConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function () {
    const result = onConnectionsChange?.apply(this, arguments);
    scheduleSync(this);
    return result;
  };
  scheduleSync(node);
}

app.registerExtension({
  name: "TuringUtils.StageBarrierOutputs",
  nodeCreated(node) {
    if (node.comfyClass === NODE_TYPE || node.constructor.type === NODE_TYPE) {
      install(node);
    }
  },
  loadedGraphNode(node) {
    if (node.comfyClass === NODE_TYPE || node.constructor.type === NODE_TYPE) {
      install(node);
      scheduleSync(node);
    }
  },
});
