import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';

interface Step {
  turn: number;
  label: string;
  order_parameter: number;
  precision_margin?: number;
}

interface Props {
  steps: Step[];
}

export default function TrajectoryChart({ steps }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !steps?.length) return;

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 24, right: 28, bottom: 48, left: 64 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current).attr('width', width).attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear()
      .domain(d3.extent(steps, (d) => d.turn) as [number, number])
      .range([0, innerWidth]);
    const y = d3.scaleLinear()
      .domain([
        Math.min(0, d3.min(steps, (d) => d.order_parameter) ?? 0),
        Math.max(1, d3.max(steps, (d) => d.order_parameter) ?? 1),
      ])
      .nice()
      .range([innerHeight, 0]);

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(steps.length).tickFormat((d) => `${d}`));
    g.append('g').call(d3.axisLeft(y));

    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--ink-faint)')
      .attr('font-size', 12)
      .text('Turn');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -48)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--ink-faint)')
      .attr('font-size', 12)
      .text('Refusal order parameter m');

    const line = d3.line<Step>()
      .x((d) => x(d.turn))
      .y((d) => y(d.order_parameter));

    g.append('path')
      .datum(steps)
      .attr('fill', 'none')
      .attr('stroke', 'var(--cyan)')
      .attr('stroke-width', 2.5)
      .attr('d', line);

    g.selectAll('.trajectory-dot')
      .data(steps)
      .enter()
      .append('circle')
      .attr('class', 'trajectory-dot')
      .attr('cx', (d) => x(d.turn))
      .attr('cy', (d) => y(d.order_parameter))
      .attr('r', 4)
      .attr('fill', 'var(--cyan)');

    g.selectAll('.trajectory-label')
      .data(steps)
      .enter()
      .append('text')
      .attr('class', 'trajectory-label')
      .attr('x', (d) => x(d.turn))
      .attr('y', (d) => y(d.order_parameter) - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--ink-faint)')
      .attr('font-size', 11)
      .text((d) => d.label);
  }, [steps]);

  return <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />;
}
