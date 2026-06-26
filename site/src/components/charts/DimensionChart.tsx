import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';

interface Props {
  data: {
    [model: string]: {
      layers: number[];
      dim: number[];
    };
  };
}

export default function DimensionChart({ data }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Process data for each model
    const models = Object.entries(data).map(([name, vals]) => ({
      name,
      layers: vals.layers,
      dims: vals.dim,
    }));

    const maxLayer = Math.max(...models.map(m => Math.max(...m.layers)));
    const maxDim = Math.max(...models.flatMap(m => m.dims));

    const xScale = d3.scaleLinear()
      .domain([0, maxLayer])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, maxDim + 2])
      .range([innerHeight, 0]);

    // Color scheme
    const colors = {
      'gemma-2-2b-it': 'var(--cyan)',
      'qwen2.5-3b-it': 'var(--violet)',
      'gemma-2-9b-it': 'var(--amber)',
    };

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 35)
      .attr('fill', 'var(--ink-faint)')
      .attr('text-anchor', 'middle')
      .text('Layer');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -45)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'var(--ink-faint)')
      .attr('text-anchor', 'middle')
      .text('Effective Dimensionality');

    // Lines
    const line = d3.line<[number, number]>()
      .x((d) => xScale(d[0]))
      .y((d) => yScale(d[1]));

    models.forEach((model) => {
      const points = model.layers.map((l, i) => [l, model.dims[i]] as [number, number]);

      g.append('path')
        .datum(points)
        .attr('fill', 'none')
        .attr('stroke', colors[model.name as keyof typeof colors] || 'var(--ink-faint)')
        .attr('stroke-width', 2)
        .attr('opacity', 0.8)
        .attr('d', line);

      // Points
      g.selectAll(`.dot-${model.name}`)
        .data(points)
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d[0]))
        .attr('cy', (d) => yScale(d[1]))
        .attr('r', 2)
        .attr('fill', colors[model.name as keyof typeof colors] || 'var(--ink-faint)')
        .attr('opacity', 0.6);
    });

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth - 200}, 10)`);

    const legendItems = [
      { name: 'Gemma-2-2B', color: 'var(--cyan)' },
      { name: 'Qwen-3B', color: 'var(--violet)' },
      { name: 'Gemma-2-9B', color: 'var(--amber)' },
    ];

    legendItems.forEach((item, i) => {
      const y = i * 18;
      legend.append('line')
        .attr('x1', 0)
        .attr('x2', 20)
        .attr('y1', y + 5)
        .attr('y2', y + 5)
        .attr('stroke', item.color)
        .attr('stroke-width', 2);

      legend.append('text')
        .attr('x', 25)
        .attr('y', y + 8)
        .attr('fill', 'var(--ink)')
        .attr('font-size', '11px')
        .text(item.name);
    });

  }, [data]);

  return (
    <svg
      ref={svgRef}
      style={{
        width: '100%',
        height: '100%',
        color: 'var(--ink)',
      }}
    />
  );
}
