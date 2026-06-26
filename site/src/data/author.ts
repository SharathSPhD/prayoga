// Canonical author + resource links (mirrors the ActiveCircuitDiscovery site).
export const AUTHOR = {
  name: 'Dr. Sharath Sathish',
  tagline: 'prayoga — refusal-suppression as a cross-domain mechanism',
  email: 'sharath.ai.colab@gmail.com',
};

export interface LinkDef { label: string; href: string; kind: 'primary' | 'social'; }

export const LINKS: LinkDef[] = [
  { label: 'Code & data', href: 'https://github.com/SharathSPhD/prayoga', kind: 'primary' },
  { label: 'HF Space (demo)', href: 'https://huggingface.co/spaces/qbz506/prayoga', kind: 'primary' },
  { label: 'HF Dataset', href: 'https://huggingface.co/datasets/qbz506/prayoga-results', kind: 'primary' },
  { label: 'Colab', href: 'https://colab.research.google.com/github/SharathSPhD/prayoga/blob/main/notebooks/00_explore_results.ipynb', kind: 'primary' },
  { label: 'TechNektar', href: 'https://www.technektar.dev/', kind: 'social' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/sharath-s', kind: 'social' },
  { label: 'Google Scholar', href: 'https://scholar.google.com/citations?hl=en&user=dcyu5ucAAAAJ', kind: 'social' },
  { label: 'ResearchGate', href: 'https://www.researchgate.net/profile/Sharath-Sathish/research', kind: 'social' },
  { label: 'Medium', href: 'https://medium.com/@sharath.ai.colab', kind: 'social' },
  { label: 'Substack', href: 'https://technektar.substack.com/', kind: 'social' },
  { label: 'YouTube', href: 'https://www.youtube.com/@SharathS-PhD', kind: 'social' },
];

export const socialLinks = LINKS.filter((l) => l.kind === 'social');
export const primaryLinks = LINKS.filter((l) => l.kind === 'primary');
