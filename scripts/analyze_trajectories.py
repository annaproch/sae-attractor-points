"""Summarize saved trajectories without rerunning or modifying the dynamics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def analyze(root):
    probes = json.loads((root / 'probes.json').read_text())
    endpoints = json.loads((root / 'states_endpoints.json').read_text())
    trajectories = {i: [] for i in range(len(probes))}
    for path in sorted((root / 'states').glob('step_*.npz')):
        step = int(path.stem.split('_')[1])
        ids = np.load(path.with_name(f'probe_ids_{step:05d}.npy'))
        for i, z in zip(ids, sparse.load_npz(path).toarray()):
            trajectories[int(i)].append(z)
    growth = [r['probe_id'] for r in endpoints if r['status'] == 'growth_limit']
    collapse = [r['probe_id'] for r in endpoints
                if r['status'] == 'fixed_point_candidate' and r['norm'] == 0 and r['probe_id']]

    def quantiles(values):
        return np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else None

    result = {'quantile_order': ['min', 'q25', 'median', 'q75', 'max'],
              'growth_count': len(growth), 'collapse_count_excluding_baseline': len(collapse)}
    result['growth_initially_shrank'] = sum(bool(np.linalg.norm(trajectories[i][1]) < np.linalg.norm(trajectories[i][0])) for i in growth)
    result['collapse_with_recruitment'] = sum(any(np.count_nonzero(z) > 1 for z in trajectories[i]) for i in collapse)
    result['steps_to_zero_quantiles'] = quantiles([next(t for t, z in enumerate(trajectories[i]) if not z.any()) for i in collapse])
    if growth:
        normalized = np.stack([trajectories[i][-1] / np.linalg.norm(trajectories[i][-1]) for i in growth])
        pair_cosines = normalized @ normalized.T
        result['growth_endpoint_pair_cosine_quantiles'] = quantiles(pair_cosines[np.triu_indices(len(growth), 1)])
        result['growth_last_step_cosine_quantiles'] = quantiles([
            float(trajectories[i][-1] @ trajectories[i][-2] / (np.linalg.norm(trajectories[i][-1]) * np.linalg.norm(trajectories[i][-2]))) for i in growth])
        result['growth_last_step_norm_ratio_quantiles'] = quantiles([np.linalg.norm(trajectories[i][-1]) / np.linalg.norm(trajectories[i][-2]) for i in growth])
        result['growth_final_l0_quantiles'] = quantiles([np.count_nonzero(trajectories[i][-1]) for i in growth])
        result['growth_original_feature_still_active'] = sum(bool(trajectories[i][-1][probes[i]['feature_id']] > 0) for i in growth)
        result['growth_original_feature_still_dominant'] = sum(bool(np.argmax(trajectories[i][-1]) == probes[i]['feature_id']) for i in growth)
        result['growth_dominant_feature_counts'] = Counter(int(np.argmax(trajectories[i][-1])) for i in growth).most_common()
    examples = []
    for feature in (120, 11341, 12692):
        matches = [i for i, p in enumerate(probes) if p.get('feature_id') == feature and p.get('scale_multiplier') == 16]
        for i in matches:
            rows = []
            for t, z in enumerate(trajectories[i]):
                nonzero = np.flatnonzero(z)
                top = nonzero[np.argsort(z[nonzero])[-5:][::-1]]
                rows.append({'step': t, 'norm': float(np.linalg.norm(z)), 'l0': len(nonzero),
                             'top_features': [{'id': int(j), 'value': float(z[j])} for j in top]})
            examples.append({'probe': probes[i], 'status': endpoints[i]['status'], 'steps': rows})
    result['examples'] = examples
    (root / 'trajectory_analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run)
