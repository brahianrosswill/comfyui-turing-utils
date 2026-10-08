import { app } from "../../scripts/app.js";
import { migrateNoiseGrid } from "./lib/node_migrations.js";

app.registerExtension({
  name: "TuringUtils.NodeConfiguration",
  nodeCreated(node) {
    if (!node.comfyClass?.startsWith("TuringUtils")) return;
    // The legacy canvas reads widget.advanced; Nodes 2.0 reads options.advanced.
    // Bridge the schema flag, keeping the frontend's own toggle and persistence.
    const sync = () => {
      for (const widget of node.widgets ?? []) {
        if (widget.options?.advanced !== undefined && widget.advanced !== widget.options.advanced) {
          widget.advanced = widget.options.advanced;
        }
      }
    };
    sync();
    const layoutWidgets = node.getLayoutWidgets;
    if (layoutWidgets && node.isWidgetVisible) {
      node.getLayoutWidgets = function (...args) {
        const widgets = layoutWidgets.apply(this, args);
        return globalThis.LiteGraph?.vueNodesMode ? widgets : widgets.filter(w => this.isWidgetVisible(w));
      };
    }
    // Size new nodes compactly; onConfigure still restores users' saved sizes.
    if (node.hasAdvancedWidgets?.()) node.setSize(node.computeSize());
    for (const method of ["onConfigure", "onWidgetChanged"]) {
      const original = node[method];
      node[method] = function (...args) {
        const result = original?.apply(this, args);
        sync();
        queueMicrotask(sync);
        return result;
      };
    }
  },
  beforeConfigureGraph(graph) {
    migrateNoiseGrid(graph);
  },
});
