"""Short, matched synthetic-background probes; no normalization during rollout."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from scripts.trajectories import SAE, save_json


def neighbor_pool(sae, targets, limit=128):
    """Union of per-target top-positive pools, ranked by best cosine then ID."""
    norms = np.linalg.norm(sae.W_dec, axis=1)
    if np.any(norms[list(targets)] == 0):
        raise ValueError('Target has zero decoder norm')
    scores = {}
    for target in targets:
        cos = np.divide(sae.W_dec @ sae.W_dec[target], norms * norms[target],
                        out=np.zeros(sae.width, dtype=np.float32), where=norms > 0)
        ids = np.flatnonzero(cos > 0)
        ids = ids[~np.isin(ids, targets)]
        ids = ids[np.lexsort((ids, -cos[ids]))][:limit]
        for rank, feature in enumerate(ids, 1):
            scores.setdefault(int(feature), {})[str(target)] = {
                'rank': rank, 'cosine': float(cos[feature])}
    ordered = sorted(scores, key=lambda f: (-max(v['cosine'] for v in scores[f].values()), f))
    return [{'feature_id': f, 'rank': i + 1, 'sources': scores[f]} for i, f in enumerate(ordered)]


def subsets(pool, sizes, repeats, seed):
    ids = [p['feature_id'] for p in pool]
    result = [{'selection': 'empty', 'features': [], 'sampling_seed': None}]
    for size in sizes:
        if size > len(ids):
            continue
        result.append({'selection': 'nearest', 'features': ids[:size], 'sampling_seed': None})
        for repeat in range(repeats):
            sample_seed = [seed, size, repeat]
            chosen = np.random.default_rng(sample_seed).choice(ids, size, replace=False)
            result.append({'selection': f'sampled_{repeat}', 'features': sorted(map(int, chosen)),
                           'sampling_seed': sample_seed})
    if ids:
        result.append({'selection': 'full', 'features': ids, 'sampling_seed': None})
    return result


def make_conditions(sae, targets, amplitudes, members, background_amplitude, mode, reference_norm):
    bg = np.zeros(sae.width, dtype=np.float32)
    bg[members] = background_amplitude
    if mode == 'matched_decoded_norm' and members:
        norm = np.linalg.norm(bg @ sae.W_dec)
        if norm <= 0:
            raise ValueError('Cannot normalize zero decoded background')
        bg *= reference_norm / norm
    a = bg.copy()
    a[targets[0]] += amplitudes[0]
    if len(targets) == 1:
        return np.stack([bg, a])
    b, ab = bg.copy(), a.copy()
    b[targets[1]] += amplitudes[1]
    ab[targets[1]] += amplitudes[1]
    return np.stack([bg, a, b, ab])


def contrast(rows):
    # Promote before subtraction so float32 cancellation does not create fake interactions.
    rows = rows.astype(np.float64)
    return rows[1] - rows[0] if len(rows) == 2 else rows[3] - rows[1] - rows[2] + rows[0]


def run(args):
    if args.steps < 1 or args.repeats < 0 or args.pool_size < 1:
        raise ValueError('Invalid steps, repeats or pool size')
    strengths = [float(v) for v in args.strengths.split(',')]
    sizes = [int(v) for v in args.sizes.split(',')]
    if not all(np.isfinite(v) and v > 0 for v in strengths + [args.background_strength]) or min(sizes) < 1:
        raise ValueError('Strengths and subset sizes must be positive')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    start = time.time()
    sae = SAE(args.checkpoint)
    families = [(120,), (11341,), (12692,), (120, 6712), (120, 2499)]
    gain = np.einsum('ij,ji->i', sae.W_dec, sae.W_enc)
    bias = sae.b_dec @ sae.W_enc + sae.b_enc
    targets = sorted({f for family in families for f in family})
    if any(gain[f] <= 0 or sae.threshold[f] <= bias[f] for f in targets):
        raise ValueError('Targets require positive isolated self-activation boundaries')
    boundary = (sae.threshold - bias) / np.where(gain > 0, gain, 1)
    groups, initial, pools, baselines = [], [], {}, []
    count = 0
    for family_index, family in enumerate(families):
        pool = neighbor_pool(sae, family, args.pool_size)
        name = '+'.join(map(str, family))
        pools[name] = pool
        bg_amplitude = float(boundary[family[0]] * args.background_strength)
        reference_norm = float(bg_amplitude * np.linalg.norm(sae.W_dec[pool[0]['feature_id']])) if pool else 0.
        for a in family:
            for b in family:
                baselines.append({'family': name, 'source': a, 'destination': b,
                    'decoder_cosine': float(sae.W_dec[a] @ sae.W_dec[b] / (np.linalg.norm(sae.W_dec[a]) * np.linalg.norm(sae.W_dec[b]))),
                    'directed_coupling': float(sae.W_dec[a] @ sae.W_enc[:, b])})
        for subset in subsets(pool, sizes, args.repeats, args.seed + family_index):
            for mode in ('equal_amplitude', 'matched_decoded_norm'):
                for strength in strengths:
                    amplitudes = [float(boundary[f] * strength) for f in family]
                    rows = make_conditions(sae, family, amplitudes, subset['features'], bg_amplitude, mode, reference_norm)
                    group = {'group_id': len(groups), 'family': name, 'targets': list(family),
                        **subset, 'size': len(subset['features']), 'mode': mode,
                        'target_strength': strength, 'target_amplitudes': amplitudes,
                        'background_amplitude_before_matching': bg_amplitude,
                        'reference_decoded_background_norm': reference_norm,
                        'probe_ids': list(range(count, count + len(rows))),
                        'condition_order': ['background', 'A'] if len(family) == 1 else ['background', 'A', 'B', 'AB'],
                        'initial_code_norms': np.linalg.norm(rows, axis=1).tolist(),
                        'initial_decoded_contribution_norms': np.linalg.norm(rows @ sae.W_dec, axis=1).tolist()}
                    groups.append(group)
                    initial.extend(rows)
                    count += len(rows)
    z = np.stack(initial)
    del initial
    initial_norms = np.linalg.norm(z, axis=1)
    save_json(root / 'groups.json', groups)
    save_json(root / 'neighbor_pools.json', pools)
    save_json(root / 'static_baselines.json', baselines)
    with args.checkpoint.open('rb') as file:
        sha = hashlib.file_digest(file, 'sha256').hexdigest()
    manifest = {'schema_version': 1, 'status': 'running', 'checkpoint': str(args.checkpoint),
        'sha256': sha, 'seed': args.seed, 'steps': args.steps, 'pool_size_per_target': args.pool_size,
        'sizes': sizes, 'sampled_subsets_per_size': args.repeats, 'target_strengths': strengths,
        'background_strength': args.background_strength, 'groups': len(groups), 'probes': len(z),
        'numpy': np.__version__, 'threads': os.environ.get('OPENBLAS_NUM_THREADS'),
        'amplitude_definition': 'Targets: isolated self-activation boundary times strength. Background: equal absolute code amplitude, anchored to A boundary times background strength.',
        'matched_norm_definition': 'Within each family, match every nonempty background to the decoded norm of its nearest single neighbor at the background amplitude.',
        'pool_union': 'Union of top-positive pools excluding all targets; descending best source cosine, feature ID breaks ties.',
        'storage': 'Every float32 state, every float64 contrast, all entries retained; fixed row IDs across steps.',
        'contrast': 'Single: A-background. Pair: AB-A-B+background.',
        'dynamics': 'Raw released-weight round trips; no intermediate normalization; fixed horizon with whole-run failure on nonfinite or 1e6-fold code norm guardrail.',
        'scope': 'Synthetic only; no semantic labels, real activations or LLM interventions.'}
    save_json(root / 'manifest.json', manifest)
    for directory in ('states', 'contrasts'):
        (root / directory).mkdir()
    np.save(root / 'states' / 'probe_ids.npy', np.arange(len(z)))
    np.save(root / 'contrasts' / 'group_ids.npy', np.arange(len(groups)))
    with (root / 'metrics.csv').open('w') as file:
        fields = ['group_id', 'step', 'contrast_norm', 'decoded_contrast_norm', 'positive_coordinates',
                  'negative_coordinates', 'max_condition_l0', 'max_condition_norm',
                  'A_target_delta', 'B_target_delta', 'A_recruited', 'A_suppressed']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for step in range(args.steps + 1):
            if not np.isfinite(z).all() or np.any(np.linalg.norm(z, axis=1) > 1e6 * np.maximum(initial_norms, 1)):
                manifest.update(status='guardrail_failed', failed_step=step)
                save_json(root / 'manifest.json', manifest)
                raise RuntimeError('Guardrail reached; no later contrasts inferred')
            sparse.save_npz(root / 'states' / f'step_{step:05d}.npz', sparse.csr_matrix(z))
            contrasts = np.stack([contrast(z[g['probe_ids']]) for g in groups])
            sparse.save_npz(root / 'contrasts' / f'step_{step:05d}.npz', sparse.csr_matrix(contrasts))
            decoded_norms = np.linalg.norm(contrasts @ sae.W_dec, axis=1)
            for g, c, dn in zip(groups, contrasts, decoded_norms):
                rows = z[g['probe_ids']]
                a = g['targets'][0]
                writer.writerow({'group_id': g['group_id'], 'step': step,
                    'contrast_norm': float(np.linalg.norm(c)), 'decoded_contrast_norm': float(dn),
                    'positive_coordinates': int(np.count_nonzero(c > 1e-6)),
                    'negative_coordinates': int(np.count_nonzero(c < -1e-6)),
                    'max_condition_l0': int(np.count_nonzero(rows, axis=1).max()),
                    'max_condition_norm': float(np.linalg.norm(rows, axis=1).max()),
                    'A_target_delta': float(rows[1, a] - rows[0, a]),
                    'B_target_delta': float(rows[2, g['targets'][1]] - rows[0, g['targets'][1]]) if len(rows) == 4 else '',
                    'A_recruited': int(np.count_nonzero((rows[1] > 0) & (rows[0] == 0))),
                    'A_suppressed': int(np.count_nonzero((rows[1] == 0) & (rows[0] > 0)))})
            file.flush()
            print(f'step {step}/{args.steps}: {len(z)} probes, max l0 {np.count_nonzero(z, axis=1).max()}', flush=True)
            if step < args.steps:
                z = sae.step(z)
    manifest.update(status='complete', elapsed_seconds=time.time() - start)
    save_json(root / 'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--pool-size', type=int, default=128)
    parser.add_argument('--sizes', default='1,4,16,64')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strengths', default='0.75,1.25')
    parser.add_argument('--background-strength', type=float, default=0.25)
    run(parser.parse_args())
