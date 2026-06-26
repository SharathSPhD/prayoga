import { useState } from 'preact/hooks';

interface Finding {
  id: string;
  title: string;
  tier: 'MECHANISM' | 'ANALOGY' | 'METAPHOR';
  verdict: 'holds' | 'supported' | 'falsified' | 'partial' | 'qualified';
  date: string;
  summary: string;
  implications: string;
}

interface Props {
  findings: Finding[];
}

export default function FindingsExplorer({ findings }: Props) {
  const [filterTier, setFilterTier] = useState<string | null>(null);
  const [filterVerdict, setFilterVerdict] = useState<string | null>(null);

  const filtered = findings.filter((f) => {
    if (filterTier && f.tier !== filterTier) return false;
    if (filterVerdict && f.verdict !== filterVerdict) return false;
    return true;
  });

  const tierColors: Record<string, string> = {
    MECHANISM: 'var(--cyan)',
    ANALOGY: 'var(--violet)',
    METAPHOR: 'var(--amber)',
  };

  const verdictColors: Record<string, string> = {
    holds: 'var(--emerald)',
    supported: 'var(--cyan)',
    falsified: 'var(--rose)',
    partial: 'var(--amber)',
    qualified: 'var(--violet)',
  };

  return (
    <div>
      {/* Filters */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
        <div>
          <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.85rem', fontWeight: 600 }}>Tier:</p>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {['MECHANISM', 'ANALOGY', 'METAPHOR'].map((tier) => (
              <button
                key={tier}
                onClick={() => setFilterTier(filterTier === tier ? null : tier)}
                style={{
                  padding: '0.4rem 0.8rem',
                  borderRadius: '4px',
                  border: 'none',
                  background: filterTier === tier ? tierColors[tier] : 'var(--bg-panel)',
                  color: filterTier === tier ? 'var(--bg)' : 'var(--ink)',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  transition: 'all 0.2s',
                }}
              >
                {tier}
              </button>
            ))}
          </div>
        </div>
        <div>
          <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.85rem', fontWeight: 600 }}>Verdict:</p>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {['holds', 'supported', 'falsified', 'partial', 'qualified'].map((v) => (
              <button
                key={v}
                onClick={() => setFilterVerdict(filterVerdict === v ? null : v)}
                style={{
                  padding: '0.4rem 0.8rem',
                  borderRadius: '4px',
                  border: 'none',
                  background: filterVerdict === v ? verdictColors[v] : 'var(--bg-panel)',
                  color: filterVerdict === v ? 'var(--bg)' : 'var(--ink)',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  transition: 'all 0.2s',
                  textTransform: 'capitalize',
                }}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
        gap: '1.5rem',
      }}>
        {filtered.map((finding) => (
          <div
            key={finding.id}
            style={{
              padding: '1.5rem',
              background: 'var(--bg-panel)',
              border: '1px solid var(--hairline)',
              borderRadius: '8px',
              transition: 'all 0.2s',
              borderLeft: `4px solid ${tierColors[finding.tier]}`,
            }}
          >
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.8rem', alignItems: 'center' }}>
              <span style={{
                display: 'inline-block',
                padding: '0.2rem 0.6rem',
                borderRadius: '3px',
                fontSize: '0.7rem',
                fontWeight: 700,
                background: tierColors[finding.tier],
                color: 'var(--bg)',
              }}>
                {finding.tier}
              </span>
              <span style={{
                display: 'inline-block',
                padding: '0.2rem 0.6rem',
                borderRadius: '3px',
                fontSize: '0.7rem',
                fontWeight: 700,
                background: verdictColors[finding.verdict],
                color: 'var(--bg)',
                textTransform: 'uppercase',
              }}>
                {finding.verdict}
              </span>
            </div>
            <h4 style={{ margin: '0.5rem 0', fontSize: '1rem' }}>{finding.id}: {finding.title}</h4>
            <p style={{ margin: '0.8rem 0 0 0', fontSize: '0.95rem', color: 'var(--ink-soft)' }}>
              {finding.summary}
            </p>
            <p style={{ margin: '0.8rem 0 0 0', fontSize: '0.9rem', color: 'var(--ink-faint)', fontStyle: 'italic' }}>
              {finding.implications}
            </p>
          </div>
        ))}
      </div>
      {filtered.length === 0 && (
        <p style={{ textAlign: 'center', color: 'var(--ink-faint)', padding: '2rem' }}>
          No findings match the selected filters.
        </p>
      )}
    </div>
  );
}
