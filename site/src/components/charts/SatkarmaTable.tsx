interface Props {
  data: Array<{
    sanskrit: string;
    effect: number;
    control: number;
    separated: boolean;
  }>;
}

export default function SatkarmaTable({ data }: Props) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--hairline)' }}>
            <th style={{ padding: '0.8rem', textAlign: 'left' }}>Act (Sanskrit)</th>
            <th style={{ padding: '0.8rem', textAlign: 'left' }}>Effect</th>
            <th style={{ padding: '0.8rem', textAlign: 'left' }}>Control</th>
            <th style={{ padding: '0.8rem', textAlign: 'left' }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--hairline)' }}>
              <td style={{ padding: '0.8rem', fontFamily: 'var(--font-serif)' }}>
                {row.sanskrit}
              </td>
              <td style={{ padding: '0.8rem' }}>
                {typeof row.effect === 'number' && row.effect.toFixed(3)}
              </td>
              <td style={{ padding: '0.8rem' }}>
                {typeof row.control === 'number' && row.control.toFixed(3)}
              </td>
              <td style={{ padding: '0.8rem' }}>
                <span
                  style={{
                    display: 'inline-block',
                    padding: '0.3rem 0.6rem',
                    borderRadius: '4px',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    background: row.separated ? 'var(--emerald)' : 'var(--rose)',
                    color: 'var(--bg)',
                  }}
                >
                  {row.separated ? '✓ Control' : '✗ No'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
