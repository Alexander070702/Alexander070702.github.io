function initComparisonChart() {
  const container = d3.select("#comparisonChart");
  const width = 300, height = 400;
  const nodeRadius = 18; // enlarge nodes

  // ─── Mock data ───
  const leftData = {
    nodes: [
      { id: 'A', x: 100, y: 50 },
      { id: 'E', x:  60, y: 150 },
      { id: 'F', x: 140, y: 150 },
      { id: 'G', x:  40, y: 250 },
      { id: 'H', x: 100, y: 250 },
      { id: 'I', x: 160, y: 250 }
    ],
    edges: [
      { source: 'A', target: 'E' },
      { source: 'E', target: 'H' }
    ]
  };
  const rightData = {
    nodes: [
      { id: 'A', x: 100, y: 50 },
      { id: 'E', x:  60, y: 150 },
      { id: 'F', x: 140, y: 150 },
      { id: 'G', x:  40, y: 250 },
      { id: 'H', x: 100, y: 250 },
      { id: 'I', x: 160, y: 250 }
    ],
    edges: [
      { source: 'A', target: 'E' },
      { source: 'A', target: 'F' },
      { source: 'E', target: 'G' },
      { source: 'E', target: 'H' },
      { source: 'F', target: 'H' },
      { source: 'F', target: 'I' }
    ]
  };

  // ─── Center data ───
  centerData(leftData, width, height);
  centerData(rightData, width, height);

  // ─── SVG containers ───
  const svgLeft  = container.append("svg").attr("width", width).attr("height", height);
  const svgRight = container.append("svg").attr("width", width).attr("height", height);

  // ─── White background ───
  [svgLeft, svgRight].forEach(svg => {
    svg.append("rect")
       .attr("x", 0).attr("y", 0)
       .attr("width", width).attr("height", height)
       .attr("fill", "#ffffff");
  });

  // ─── Arrow markers ───
  [svgLeft, svgRight].forEach(svg => {
    const defs = svg.append("defs");

    // darker arrow used normally
    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", nodeRadius + 6)
      .attr("refY", 0)
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .attr("orient", "auto")
      .append("path")
        .attr("d", "M0,-5 L10,0 L0,5 Z")
        .attr("fill", "#000");

    // lighter arrow for highlighting steps
    defs.append("marker")
      .attr("id", "arrow-soft")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", nodeRadius + 6)
      .attr("refY", 0)
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .attr("orient", "auto")
      .append("path")
        .attr("d", "M0,-5 L10,0 L0,5 Z")
        .attr("fill", "#444");
  });

  // ─── Draw DAGs ───
  const edgesLeft  = drawDAG(svgLeft,  leftData,  "Traditional");
  const edgesRight = drawDAG(svgRight, rightData, "GFlowNet");

  // ─── Animation ───
  let step = 0;
  d3.interval(() => {
    step = (step + 1) % 6;

    // Traditional highlights
    edgesLeft.transition().duration(600)
      .attr("stroke-width", d => {
        const act = (step < 3 && d.source==='A'&&d.target==='E') ||
                    (step>=3 && d.source==='E'&&d.target==='H');
        return act ? 4 : 2;
      })
      .attr("stroke", d => {
        const act = (step < 3 && d.source==='A'&&d.target==='E') ||
                    (step>=3 && d.source==='E'&&d.target==='H');
        return act ? "#ff9999" : "#000";
      })
      .attr("marker-end", d => {
        const act = (step < 3 && d.source==='A'&&d.target==='E') ||
                    (step>=3 && d.source==='E'&&d.target==='H');
        return `url(#${act ? 'arrow-soft' : 'arrow'})`;
      });

    // GFlowNet highlights
    edgesRight.transition().duration(600)
      .attr("stroke-width", step>=3 ? 3 : 2)
      .attr("stroke", step>=3 ? "#66c2ff" : "#000")
      .attr("marker-end", step>=3 ? "url(#arrow-soft)" : "url(#arrow)");

  }, 1800);


  // ─── Helpers ───
  function centerData(data, w, h) {
    const xs = data.nodes.map(n=>n.x), ys = data.nodes.map(n=>n.y);
    const minX = d3.min(xs), maxX = d3.max(xs);
    const minY = d3.min(ys), maxY = d3.max(ys);
    const dx = maxX - minX, dy = maxY - minY;
    const offX = (w - dx)/2 - minX, offY = (h - dy)/2 - minY;
    data.nodes.forEach(n=>{ n.x += offX; n.y += offY; });
  }

  function drawDAG(svg, data, title) {
    // title
    svg.append("text")
      .attr("x", width/2).attr("y", 24)
      .attr("text-anchor", "middle")
      .attr("fill", "#000")
      .attr("font-size", 18).attr("font-weight", "bold")
      .text(title);

    // edges
    const edges = svg.selectAll(".edge")
      .data(data.edges).enter().append("line")
        .attr("class", "edge")
        .attr("x1", d=>data.nodes.find(n=>n.id===d.source).x)
        .attr("y1", d=>data.nodes.find(n=>n.id===d.source).y)
        .attr("x2", d=>data.nodes.find(n=>n.id===d.target).x)
        .attr("y2", d=>data.nodes.find(n=>n.id===d.target).y)
        .attr("stroke", "#000")
        .attr("stroke-width", 2)
        .attr("marker-end", "url(#arrow)");

    // nodes
    svg.selectAll(".node")
      .data(data.nodes).enter().append("circle")
        .attr("class", "node")
        .attr("r", nodeRadius)
        .attr("cx", d=>d.x)
        .attr("cy", d=>d.y)
        .attr("fill", "#1f77b4")
        .attr("stroke", "#fff")
        .attr("stroke-width", 2);

    // labels
    svg.selectAll(".label")
      .data(data.nodes).enter().append("text")
        .attr("class", "label")
        .attr("x", d=>d.x)
        .attr("y", d=>d.y+4)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .attr("font-size", 12)
        .attr("font-weight", "bold")
        .text(d=>d.id);

    return edges;
  }
}
