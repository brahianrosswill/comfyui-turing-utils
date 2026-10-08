import assert from 'node:assert/strict';
import { test } from 'node:test';
import { migrateNoiseGrid } from '../web/lib/node_migrations.js';

test('flat grid widgets migrate without changing links or active block values', () => {
  const node={id:4,type:'TuringUtilsVideoPrefixContextNoise',widgets_values:[5,.45,0,'fixed',.1,4,'gaussian_rgb','block_size',7],
    inputs:[{name:'block_size',type:'INT',link:81,widget:{name:'block_size'}}],
    widgets_values_named:{grid_mode:'block_size',block_size:7}};
  const graph={nodes:[node],links:[[81,2,0,4,0,'INT']]};
  const links=structuredClone(graph.links);
  migrateNoiseGrid(graph);
  assert.deepEqual(graph.links,links);
  assert.equal(node.inputs[0].name,'grid_mode.block_size');
  assert.equal(node.inputs[0].link,81);
  assert.equal(node.widgets_values.at(-1),7);
  assert.equal(node.widgets_values_named['grid_mode.block_size'],7);
  const once=structuredClone(graph);
  assert.deepEqual(migrateNoiseGrid(graph),once);
});

test('subgraph definitions migrate and fixed grids drop only the obsolete widget', () => {
  const node={id:4,type:'TuringUtilsVideoPrefixContextNoise',widgets_values:[5,.45,0,'fixed',.1,4,'poc_chroma_blocks','poc_36x64',7]};
  const workflow={nodes:[],definitions:{subgraphs:[{nodes:[node]}]}};
  migrateNoiseGrid(workflow);
  assert.equal(node.widgets_values.length,8);
  assert.equal(node.widgets_values[7],'poc_36x64');
});
