import { app } from "../../scripts/app.js";

const NODE_TYPE = "TuringUtilsH3KeyframeReference";
const OUTPUT_TYPE = "TURING_UTILS_H3_KEYFRAME_REFERENCE";
const IMAGE_INPUT = /^images\.image_\d+$/;
const installed = new WeakSet();

function syncOutputs(node) {
  const outputCount = node.inputs.filter((input) => IMAGE_INPUT.test(input.name)).length;

  while (node.outputs.length > outputCount) {
    node.removeOutput(node.outputs.length - 1);
  }
  while (node.outputs.length < outputCount) {
    const index = node.outputs.length;
    node.addOutput(`keyframe_${index}`, OUTPUT_TYPE);
  }
  node.outputs.forEach((output, index) => {
    output.name = `keyframe_${index}`;
    output.type = OUTPUT_TYPE;
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
  name: "TuringUtils.H3KeyframeOutputs",
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
