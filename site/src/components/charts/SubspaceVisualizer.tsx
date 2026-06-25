interface ModelSubspace {
  model: string;
  layer: string;
  effectiveDim: number;
  qualifier: string;
}

interface Props {
  models: ModelSubspace[];
}

export default function SubspaceVisualizer({ models }: Props) {
  const maxDim = Math.max(...models.map((m) => m.effectiveDim), 1);

  return (
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      {models.map((m) => (
        <div class="bg-slate-800/40 border border-slate-700 rounded-xl p-5">
          <div class="text-sm text-slate-400 mb-2">{m.layer}</div>
          <h4 class="text-lg font-bold text-slate-100 mb-4">{m.model}</h4>
          <div class="h-3 bg-slate-700 rounded-full overflow-hidden mb-3">
            <div
              class="h-full bg-cyan-400"
              style={{ width: `${Math.max(8, (100 * m.effectiveDim) / maxDim)}%` }}
            />
          </div>
          <div class="text-3xl font-bold text-cyan-300 mb-3">{m.effectiveDim}D</div>
          <p class="text-sm text-slate-300">{m.qualifier}</p>
        </div>
      ))}
    </div>
  );
}
