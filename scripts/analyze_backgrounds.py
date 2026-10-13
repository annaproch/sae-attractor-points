"""Summarize saved background probes without running further SAE steps."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.trajectories import save_json


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Analysis requires a complete run')
    groups = json.loads((root / 'groups.json').read_text())
    summaries = []
    survivors = []
    for step in range(1, manifest['steps'] + 1):
        z = sparse.load_npz(root / 'states' / f'step_{step:05d}.npz').toarray()
        contrasts = sparse.load_npz(root / 'contrasts' / f'step_{step:05d}.npz').toarray()
        norms = np.linalg.norm(contrasts, axis=1)
        row = {'step': step, 'nonzero_probes': int(np.any(z, axis=1).sum()),
               'active_feature_ids': np.flatnonzero(np.any(z, axis=0)).tolist(), 'by_mode': {}}
        for mode in ('equal_amplitude', 'matched_decoded_norm'):
            selected = [g for g in groups if g['mode'] == mode]
            ids = [i for g in selected for i in g['probe_ids']]
            row['by_mode'][mode] = {
                'nonzero_probes': int(np.any(z[ids], axis=1).sum()),
                'nonzero_pair_contrasts': sum(bool(norms[g['group_id']] > 1e-6)
                                              for g in selected if len(g['targets']) == 2)}
        summaries.append(row)
        if step == manifest['steps']:
            for g in groups:
                if not np.any(z[g['probe_ids']]):
                    continue
                detail = {k: g[k] for k in ('group_id', 'family', 'size', 'selection', 'mode',
                                           'target_strength', 'initial_decoded_contribution_norms')}
                detail['conditions'] = {name: {str(i): float(values[i]) for i in np.flatnonzero(values)}
                                        for name, values in zip(g['condition_order'], z[g['probe_ids']])}
                survivors.append(detail)
    result = {'contrast_nonzero_tolerance': 1e-6, 'steps': summaries, 'final_survivors': survivors,
              'caveat': 'Counts include reused or duplicate controls; probes are not independent samples.'}
    save_json(root / 'analysis.json', result)
    print(json.dumps({'steps': [(r['step'], r['nonzero_probes']) for r in summaries],
                      'final_surviving_groups': len(survivors)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run)
