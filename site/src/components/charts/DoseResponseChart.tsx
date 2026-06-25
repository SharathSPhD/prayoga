import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';

interface Props {
  data: {
    alphas: number[];
    asr_real: number[];
    asr_random: number[];
    ec50: number;
    r2: number;
  };
}

export default function DoseResponseChart({ data }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 20, right: 30, bottom: 30, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);

    // Data points
    const realPoints = data.alphas.map((d, i) => ({
      alpha: d,
      asr: data.asr_real[i],
    }));

    const randomPoints = data.alphas.map((d, i) => ({
      alpha: d,
      asr: data.asr_random[i],
    }));

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 35)
      .attr('fill', 'var(--ink-faint)')
      .attr('text-anchor', 'middle')
      .text('Ablation fraction (α)');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -45)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'var(--ink-faint)')
      .attr('text-anchor', 'middle')
      .text('Attack Success Rate (ASR)');

    // Real curve
    const line = d3.line<typeof realPoints[0]>()
      .x((d) => xScale(d.alpha))
      .y((d) => yScale(d.asr));

    g.append('path')
      .datum(realPoints)
      .attr('fill', 'none')
      .attr('stroke', 'var(--cyan)')
      .attr('stroke-width', 2.5)
      .attr('d', line);

    // Real points
    g.selectAll('.dot-real')
      .data(realPoints)
      .enter()
      .append('circle')
      .attr('class', 'dot-real')
      .attr('cx', (d) => xScale(d.alpha))
      .attr('cy', (d) => yScale(d.asr))
      .attr('r', 3)
      .attr('fill', 'var(--cyan)');

    // Random line
    g.append('path')
      .datum(randomPoints)
      .attr('fill', 'none')
      .attr('stroke', 'var(--ink-faint)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,4')
      .attr('d', line);

    // EC50 marker
    g.append('line')
      .attr('x1', xScale(data.ec50))
      .attr('x2', xScale(data.ec50))
      .attr('y1', 0)
      .attr('y2', innerHeight)
      .attr('stroke', 'var(--amber)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '2,2')
      .attr('opacity', 0.5);

    g.append('text')
      .attr('x', xScale(data.ec50))
      .attr('y', -5)
      .attr('fill', 'var(--amber)')
      .attr('font-size', '12px')
      .attr('text-anchor', 'middle')
      .text(`EC50=${data.ec50.toFixed(3)}`);

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth - 150}, 10)`);

    legend.append('rect')
      .attr('width', 140)
      .attr('height', 50)
      .attr('fill', 'var(--bg-inset)')
      .attr('stroke', 'var(--hairline)')
      .attr('rx', 4);

    legend.append('line')
      .attr('x1', 10)
      .attr('x2', 30)
      .attr('y1', 15)
      .attr('y2', 15)
      .attr('stroke', 'var(--cyan)')
      .attr('stroke-width', 2);

    legend.append('text')
      .attr('x', 40)
      .attr('y', 18)
      .attr('fill', 'var(--ink)')
      .attr('font-size', '11px')
      .text('Ablation');

    legend.append('line')
      .attr('x1', 10)
      .attr('x2', 30)
      .attr('y1', 33)
      .attr('y2', 33)
      .attr('stroke', 'var(--ink-faint)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,4');

    legend.append('text')
      .attr('x', 40)
      .attr('y', 36)
      .attr('fill', 'var(--ink-faint)')
      .attr('font-size', '11px')
      .text('Random');

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
