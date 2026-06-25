import { useEffect, useRef } from 'preact/hooks';
import * as d3 from 'd3';

interface Props {
  data: {
    [model: string]: {
      F_refusal: number;
      F_random: number;
      m_plain: number;
      m_injected: number;
    };
  };
}

export default function SymmetryChart({ data }: Props) {
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

    // Data for grouped bar chart
    const models = Object.entries(data).map(([name, vals]) => ({
      name: name === 'gemma-2-2b-it' ? 'Gemma-2B' : 'Qwen-3B',
      F_refusal: vals.F_refusal,
      F_random: vals.F_random,
      m_plain: vals.m_plain,
      m_injected: vals.m_injected,
    }));

    const metrics = [
      { key: 'F_refusal', label: 'F-ratio (Refusal)', yMax: 25 },
      { key: 'F_random', label: 'F-ratio (Random)', yMax: 25 },
      { key: 'm_plain', label: 'Order Param (Plain)', yMax: 0.35 },
      { key: 'm_injected', label: 'Order Param (Injected)', yMax: 0.35 },
    ];

    // Create subgroups
    const xScale = d3.scaleBand()
      .domain(models.map(m => m.name))
      .range([0, innerWidth])
      .padding(0.3);

    const subgroups = d3.scaleBand()
      .domain(metrics.map(m => m.key))
      .range([0, xScale.bandwidth()])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, 25])
      .range([innerHeight, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 35)
      .attr('fill', 'var(--ink-faint)')
      .attr('text-anchor', 'middle')
      .text('Model');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -45)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'var(--ink-faint)')
      .attr('text-anchor', 'middle')
      .text('F-ratio / Order Parameter');

    // Colors for metrics
    const colors: Record<string, string> = {
      F_refusal: 'var(--cyan)',
      F_random: 'var(--ink-faint)',
      m_plain: 'var(--violet)',
      m_injected: 'var(--rose)',
    };

    // Bars
    models.forEach((model) => {
      const modelGroup = g.append('g')
        .attr('transform', `translate(${xScale(model.name)},0)`);

      metrics.forEach((metric) => {
        const value = model[metric.key as keyof typeof model] as number;
        const scaledValue = metric.yMax === 0.35 ? value * 70 : value; // Normalize for display

        modelGroup.append('rect')
          .attr('x', subgroups(metric.key))
          .attr('y', yScale(scaledValue))
          .attr('width', subgroups.bandwidth())
          .attr('height', innerHeight - yScale(scaledValue))
          .attr('fill', colors[metric.key])
          .attr('opacity', 0.8);
      });
    });

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth - 200}, 10)`);

    const legendMetrics = [
      { key: 'F_refusal', label: 'F-refusal' },
      { key: 'F_random', label: 'F-random' },
      { key: 'm_plain', label: 'm-plain' },
      { key: 'm_injected', label: 'm-injected' },
    ];

    legendMetrics.forEach((item, i) => {
      const y = i * 18;
      legend.append('rect')
        .attr('x', 0)
        .attr('y', y - 5)
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', colors[item.key]);

      legend.append('text')
        .attr('x', 18)
        .attr('y', y + 3)
        .attr('fill', 'var(--ink)')
        .attr('font-size', '10px')
        .text(item.label);
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
