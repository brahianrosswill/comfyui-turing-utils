const NOISE = "TuringUtilsVideoPrefixContextNoise";

// Keep slot indices/link IDs intact; only the widget's dynamic path changes.
export function migrateNoiseGrid(workflow) {
  const graphs = [workflow, ...(workflow.definitions?.subgraphs ?? [])];
  for (const graph of graphs) {
    const migrated = new Set();
    for (const node of graph.nodes ?? []) {
      if (node.type !== NOISE) continue;
      if (node.properties?.turing_noise_grid_version === 1) continue;
      migrated.add(String(node.id));
      for (const input of node.inputs ?? []) {
        if (input.name !== "block_size") continue;
        input.name = "grid_mode.block_size";
        if (input.widget) input.widget.name = "grid_mode.block_size";
      }
      const named = node.widgets_values_named;
      if (named && "block_size" in named) {
        named["grid_mode.block_size"] = named.block_size;
        delete named.block_size;
      }
      // In the old layout block_size was the final widget, even in fixed-grid mode.
      if (Array.isArray(node.widgets_values) && node.widgets_values.length === 9 && node.widgets_values[7] === "poc_36x64") {
        node.widgets_values.pop();
      }
      node.properties ??= {};
      node.properties.turing_noise_grid_version = 1;
    }
    // Legacy subgraph/widget-proxy representations reference widgets by name.
    for (const node of graph.nodes ?? []) {
      for (const proxy of node.properties?.proxyWidgets ?? []) {
        if (migrated.has(String(proxy[0])) && proxy[1] === "block_size") proxy[1] = "grid_mode.block_size";
      }
    }
  }
  return workflow;
}
