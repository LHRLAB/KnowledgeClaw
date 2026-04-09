/**
 * KnowledgeClaw – D3 v7 Force-Directed Graph Visualisation
 *
 * Fetches /api/graph -> {nodes: [{id, label, type, degree, source}], links: [{source, target}]}
 * Renders a force simulation with:
 *   - Nodes coloured by type (entity=indigo, hyperedge=orange, unknown=grey)
 *   - Node radius scaled by degree
 *   - Drag, zoom, click-to-inspect
 *   - Text filter
 */

'use strict';

// ---- Colour palette ----
const NODE_COLOURS = {
  entity:     '#6366f1',
  hyperedge:  '#f97316',
  unknown:    '#6b7280',
};

const STROKE_COLOURS = {
  entity:     '#818cf8',
  hyperedge:  '#fb923c',
  unknown:    '#9ca3af',
};

// ---- Global state ----
let allNodes = [];
let allLinks = [];
let simulation = null;
let svg = null;
let mainGroup = null;
let linkSelection = null;
let nodeSelection = null;
let currentFilter = '';
let zoomBehavior = null;

// ---- Radius helpers ----
function nodeRadius(d) {
  const base = d.type === 'hyperedge' ? 5 : 6;
  const extra = Math.min(Math.sqrt(d.degree || 0) * 1.5, 14);
  return base + extra;
}

// ---- Init ----
function initGraph() {
  const canvasEl = document.getElementById('graph-canvas');
  const width = canvasEl.clientWidth || window.innerWidth - 300;
  const height = canvasEl.clientHeight || window.innerHeight - 100;

  svg = d3.select('#graph-canvas')
    .attr('width', width)
    .attr('height', height);

  svg.selectAll('*').remove();

  // Arrow marker
  svg.append('defs').append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 18)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', 'var(--border)');

  mainGroup = svg.append('g').attr('class', 'main-group');

  zoomBehavior = d3.zoom()
    .scaleExtent([0.05, 8])
    .on('zoom', (event) => mainGroup.attr('transform', event.transform));

  svg.call(zoomBehavior);

  // Empty containers for links and nodes (nodes rendered on top)
  mainGroup.append('g').attr('class', 'links-layer');
  mainGroup.append('g').attr('class', 'nodes-layer');
}

// ---- Load & render ----
async function loadGraph() {
  try {
    const data = await apiFetch('/api/graph');
    allNodes = data.nodes || [];
    allLinks = data.links || [];

    document.getElementById('graphStats').textContent =
      `${allNodes.length} nodes · ${allLinks.length} edges`;

    renderGraph(allNodes, allLinks);
    if (typeof currentLang === 'function' && currentLang() === 'zh') applyChineseLabels();
  } catch (err) {
    console.error('Failed to load graph:', err);
  }
}

function renderGraph(nodes, links) {
  const canvasEl = document.getElementById('graph-canvas');
  const width  = canvasEl.clientWidth  || 900;
  const height = canvasEl.clientHeight || 600;

  // Update SVG dimensions
  svg.attr('width', width).attr('height', height);

  // Build index maps for D3 link resolution
  const nodeById = new Map(nodes.map(n => [n.id, n]));

  // Resolve link objects
  const resolvedLinks = links
    .map(l => ({
      source: typeof l.source === 'object' ? l.source : nodeById.get(l.source),
      target: typeof l.target === 'object' ? l.target : nodeById.get(l.target),
    }))
    .filter(l => l.source && l.target);

  // ---- Simulation ----
  if (simulation) simulation.stop();

  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(resolvedLinks)
      .id(d => d.id)
      .distance(d => {
        const src = d.source;
        const tgt = d.target;
        if (src && tgt && src.type === 'hyperedge' && tgt.type === 'entity') return 70;
        return 90;
      })
      .strength(0.5))
    .force('charge', d3.forceManyBody()
      .strength(d => d.type === 'hyperedge' ? -120 : -80)
      .distanceMax(300))
    .force('center', d3.forceCenter(width / 2, height / 2).strength(0.08))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 4).strength(0.7))
    .alphaDecay(0.025)
    .velocityDecay(0.4);

  // ---- Links ----
  const linksLayer = mainGroup.select('.links-layer');
  linksLayer.selectAll('*').remove();

  linkSelection = linksLayer.selectAll('.link')
    .data(resolvedLinks, d => `${d.source.id ?? d.source}--${d.target.id ?? d.target}`)
    .join('line')
    .attr('class', 'link')
    .attr('marker-end', 'url(#arrow)');

  // ---- Nodes ----
  const nodesLayer = mainGroup.select('.nodes-layer');
  nodesLayer.selectAll('*').remove();

  nodeSelection = nodesLayer.selectAll('.node')
    .data(nodes, d => d.id)
    .join('g')
    .attr('class', 'node')
    .call(d3.drag()
      .on('start', dragStarted)
      .on('drag', dragged)
      .on('end', dragEnded))
    .on('click', (event, d) => {
      event.stopPropagation();
      showDetails(d, resolvedLinks);
    });

  nodeSelection.append('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => NODE_COLOURS[d.type] || NODE_COLOURS.unknown)
    .attr('stroke', d => STROKE_COLOURS[d.type] || STROKE_COLOURS.unknown)
    .attr('stroke-width', 1.5)
    .attr('opacity', 0.92);

  // Label (only show for high-degree or when few nodes)
  nodeSelection.append('text')
    .attr('dy', d => nodeRadius(d) + 12)
    .attr('text-anchor', 'middle')
    .attr('font-size', '10px')
    .text(d => {
      if (nodes.length > 200 && d.degree < 3) return '';
      const max = 22;
      return d.label.length > max ? d.label.slice(0, max) + '…' : d.label;
    });

  // Click on blank canvas to close details
  svg.on('click', () => closeDetails());

  // ---- Tick ----
  simulation.on('tick', () => {
    linkSelection
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    nodeSelection.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Apply current filter if any
  if (currentFilter) filterNodes(currentFilter);
}

// ---- Drag ----
function dragStarted(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}
function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}
function dragEnded(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}

// ---- Zoom control ----
function resetZoom() {
  const canvasEl = document.getElementById('graph-canvas');
  const w = canvasEl.clientWidth || 900;
  const h = canvasEl.clientHeight || 600;
  svg.transition().duration(500).call(
    zoomBehavior.transform,
    d3.zoomIdentity.translate(w / 2, h / 2).scale(0.9).translate(-w / 2, -h / 2)
  );
}

// ---- Filter ----
function filterNodes(query) {
  currentFilter = query.toLowerCase().trim();
  if (!nodeSelection) return;

  nodeSelection.each(function(d) {
    const match = !currentFilter || d.label.toLowerCase().includes(currentFilter);
    d3.select(this).style('opacity', match ? 1 : 0.12);
  });

  if (linkSelection) {
    linkSelection.style('opacity', d => {
      if (!currentFilter) return 0.6;
      const srcMatch = d.source.label && d.source.label.toLowerCase().includes(currentFilter);
      const tgtMatch = d.target.label && d.target.label.toLowerCase().includes(currentFilter);
      return (srcMatch || tgtMatch) ? 0.8 : 0.08;
    });
  }
}

// ---- Details panel ----
let _activeNode = null;
let _activeLinks = null;

function _displayLabel(node) {
  const lang = typeof currentLang === 'function' ? currentLang() : 'en';
  if (lang === 'zh' && node.name_zh && node.name_zh !== node.label) return node.name_zh;
  return node.label;
}

function showDetails(node, links) {
  _activeNode = node;
  _activeLinks = links;
  _renderDetails();

  if (nodeSelection) nodeSelection.classed('highlighted', d => d.id === node.id);
  if (linkSelection) {
    linkSelection.classed('highlighted', l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      return sid === node.id || tid === node.id;
    });
  }
}

function _renderDetails() {
  if (!_activeNode) return;
  const node = _activeNode;
  const links = _activeLinks;

  document.getElementById('details-panel').classList.add('open');
  document.getElementById('detail-label').textContent = _displayLabel(node);
  document.getElementById('detail-source').textContent = node.source || '–';
  document.getElementById('detail-degree').textContent = node.degree;

  const badge = document.getElementById('detail-type-badge');
  badge.textContent = (typeof t === 'function') ? t('graph.legend.' + node.type) : node.type;
  badge.className = `detail-badge badge-${node.type}`;

  const connected = (links || [])
    .filter(l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      return sid === node.id || tid === node.id;
    })
    .map(l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      return sid === node.id
        ? (typeof l.target === 'object' ? l.target : { id: l.target, label: l.target })
        : (typeof l.source === 'object' ? l.source : { id: l.source, label: l.source });
    });

  const connDiv = document.getElementById('detail-connections');
  if (connected.length === 0) {
    connDiv.textContent = (typeof t === 'function') ? t('graph.details.none') : 'No connections';
  } else {
    connDiv.innerHTML = connected.slice(0, 30).map(c =>
      `<div style="padding:3px 0; border-bottom:1px solid var(--border);">${escapeHtml(_displayLabel(c))}</div>`
    ).join('') + (connected.length > 30
      ? `<div style="color:var(--text-muted); margin-top:4px;">…+${connected.length - 30}</div>` : '');
  }
}

// Refresh open panel when language switches
document.addEventListener('localechange', () => { if (_activeNode) _renderDetails(); });

function closeDetails() {
  _activeNode = null;
  _activeLinks = null;
  document.getElementById('details-panel').classList.remove('open');
  if (nodeSelection) nodeSelection.classed('highlighted', false);
  if (linkSelection) linkSelection.classed('highlighted', false);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ---- Node label language switch ----
// Chinese names (name_zh) are embedded in node data at ingestion time.
// No runtime translation API calls are made.

function _applyLabelTransform(transformFn) {
  if (!nodeSelection) return;
  nodeSelection.select('text').text(d => {
    const label = transformFn(d);
    if (allNodes.length > 200 && d.degree < 3) return '';
    const max = 22;
    return label.length > max ? label.slice(0, max) + '…' : label;
  });
}

function applyChineseLabels() {
  _applyLabelTransform(d => (d.name_zh && d.name_zh !== d.label ? d.name_zh : d.label));
}

function restoreOriginalLabels() {
  _applyLabelTransform(d => d.label);
}

document.addEventListener('localechange', ({ detail }) => {
  if (detail.lang === 'zh') applyChineseLabels();
  else restoreOriginalLabels();
});

// ---- Resize observer ----
const resizeObserver = new ResizeObserver(() => {
  if (svg && simulation) {
    const el = document.getElementById('graph-canvas');
    const w = el.clientWidth;
    const h = el.clientHeight;
    svg.attr('width', w).attr('height', h);
    simulation.force('center', d3.forceCenter(w / 2, h / 2).strength(0.08));
    simulation.alpha(0.2).restart();
  }
});

// ---- Bootstrap ----
document.addEventListener('DOMContentLoaded', () => {
  initGraph();
  loadGraph();
  const canvas = document.getElementById('graph-canvas');
  if (canvas) resizeObserver.observe(canvas);
});
