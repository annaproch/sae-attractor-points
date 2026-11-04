"""CPU Gemma Scope round trips. Raw released parameters need no extra centering.

Sparse files preserve ALL nonzero float32 entries, including the initial state.
No normalization or top-k truncation is applied during iteration.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy import sparse


class SAE:
    def __init__(self, path):
        if Path(path).suffix == '.safetensors':
            from safetensors.numpy import load_file
            params = load_file(str(path))
            for key in ('W_enc', 'W_dec', 'b_enc', 'b_dec', 'threshold'):
                setattr(self, key, np.asarray(params[key.lower()], dtype=np.float32))
        else:
            with np.load(path, allow_pickle=False) as params:
                for key in ('W_enc', 'W_dec', 'b_enc', 'b_dec', 'threshold'):
                    setattr(self, key, np.asarray(params[key], dtype=np.float32))
        self.width, self.dim = self.W_dec.shape
        assert self.W_enc.shape == (self.dim, self.width)
        assert self.b_enc.shape == self.threshold.shape == (self.width,)
        assert self.b_dec.shape == (self.dim,)
        assert np.all(self.threshold > 0)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        return np.where(pre > self.threshold, np.maximum(pre, 0), 0)

    def step(self, z):
        return self.encode(self.decode(z))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def rollout(sae, initial, output, steps, atol=1e-5, rtol=1e-5):
    """Save every state; stop each trajectory independently after five small steps."""
    output.mkdir(parents=True, exist_ok=False)
    z = initial.copy()
    n = len(z)
    active = np.ones(n, dtype=bool)
    streak = np.zeros(n, dtype=int)
    final_step = np.zeros(n, dtype=int)
    status = np.full(n, 'max_steps', dtype=object)
    initial_norm = np.linalg.norm(z, axis=1)
    previous = None
    residuals = np.full(n, np.nan)
    with (output.parent / (output.name + '_steps.csv')).open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['probe_id', 'step', 'l0', 'code_norm', 'step_distance',
                         'relative_step', 'decoded_step_distance', 'distance_from_initial',
                         'support_changes'])
        sparse.save_npz(output / 'step_00000.npz', sparse.csr_matrix(z))
        np.save(output / 'probe_ids_00000.npy', np.arange(n))
        for i in range(n):
            writer.writerow([i, 0, np.count_nonzero(z[i]), float(initial_norm[i]),
                             '', '', '', 0, 0])
        for t in range(1, steps + 1):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            old = z[ids].copy()
            with np.errstate(over='ignore', invalid='ignore'):
                new = sae.step(old)
                delta = np.linalg.norm(new - old, axis=1)
                norm = np.linalg.norm(new, axis=1)
                decoded_delta = np.linalg.norm((new - old) @ sae.W_dec, axis=1)
            residuals[ids] = delta
            sparse.save_npz(output / f'step_{t:05d}.npz', sparse.csr_matrix(new))
            np.save(output / f'probe_ids_{t:05d}.npy', ids)
            for k, i in enumerate(ids):
                writer.writerow([i, t, np.count_nonzero(new[k]), float(norm[k]),
                                 float(delta[k]), float(delta[k] / max(np.linalg.norm(old[k]), 1e-12)),
                                 float(decoded_delta[k]), float(np.linalg.norm(new[k]-initial[i])),
                                 np.count_nonzero((new[k] != 0) != (old[k] != 0))])
            small = delta <= atol + rtol * np.linalg.norm(old, axis=1)
            streak[ids] = np.where(small, streak[ids] + 1, 0)
            for k, i in enumerate(ids):
                reason = None
                if not np.isfinite(new[k]).all():
                    reason = 'non_finite'
                elif norm[k] > 1e6 * max(initial_norm[i], 1):
                    reason = 'growth_limit'
                elif streak[i] >= 5:
                    reason = 'fixed_point_candidate'
                elif previous is not None and not small[k]:
                    if np.linalg.norm(new[k]-previous[i]) <= atol + rtol*norm[k]:
                        reason = 'period_2_candidate'
                if reason:
                    status[i] = reason
                    active[i] = False
                final_step[i] = t
            previous = z.copy()
            z[ids] = new
            file.flush()
            if t % 25 == 0:
                print(json.dumps({'step': t, 'active': int(active.sum()),
                                  'median_l0': float(np.median(np.count_nonzero(new, axis=1)))}), flush=True)
    sparse.save_npz(output.parent / (output.name + '_endpoints.npz'), sparse.csr_matrix(z))
    # Fresh residuals avoid declaring an endpoint from only its previous step size.
    finite = np.isfinite(z).all(axis=1)
    residuals[finite] = np.linalg.norm(sae.step(z[finite]) - z[finite], axis=1)
    records = []
    for i in range(n):
        records.append({'probe_id': i, 'status': status[i], 'final_step': int(final_step[i]),
                        'l0': int(np.count_nonzero(z[i])),
                        'norm': float(np.linalg.norm(z[i])) if finite[i] else None,
                        'fixed_residual': float(residuals[i]) if np.isfinite(residuals[i]) else None})
    save_json(output.parent / (output.name + '_endpoints.json'), records)
    return z, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--features', type=int, default=32)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--revision', default='unknown')
    parser.add_argument('--multipliers', default='1.25,2,4')
    parser.add_argument('--amplitude-mode', choices=['threshold', 'self_activation'], default='threshold')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    start = time.time()
    sae = SAE(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    ids = np.sort(rng.choice(sae.width, min(args.features, sae.width), replace=False))
    probes = [{'probe_id': 0, 'feature_id': None, 'threshold_multiplier': 0, 'amplitude': 0}]
    multipliers = [float(x) for x in args.multipliers.split(',')]
    scale = sae.threshold.copy()
    if args.amplitude_mode == 'self_activation':
        gain = np.einsum('ij,ji->i', sae.W_dec, sae.W_enc)
        if np.any(gain <= 0):
            raise ValueError('Self-activation calibration requires positive diagonal gain')
        scale = np.maximum((sae.threshold - (sae.b_dec @ sae.W_enc + sae.b_enc)) / gain, 0)
    initial = np.zeros((1 + len(ids)*len(multipliers), sae.width), dtype=np.float32)
    for feature in ids:
        for multiplier in multipliers:
            i = len(probes)
            amp = float(scale[feature] * multiplier)
            initial[i, feature] = amp
            probes.append({'probe_id': i, 'feature_id': int(feature),
                           'scale_multiplier': multiplier, 'amplitude_mode': args.amplitude_mode, 'amplitude': amp})
    save_json(root / 'probes.json', probes)
    manifest = {'schema_version': 1, 'checkpoint': str(Path(args.checkpoint).resolve()),
                'sha256': hashlib.file_digest(open(args.checkpoint, 'rb'), 'sha256').hexdigest(),
                'repository': 'google/gemma-scope-2b-pt-res', 'revision': args.revision,
                'checkpoint_file': 'layer_12/width_16k/average_l0_82/params.npz',
                'width': sae.width, 'activation_dim': sae.dim,
                'architecture': 'JumpReLU', 'dtype': 'float32', 'seed': args.seed,
                'numpy': np.__version__, 'python': platform.python_version(),
                'threads': os.environ.get('OPENBLAS_NUM_THREADS'),
                'steps': args.steps, 'atol': 1e-5, 'rtol': 1e-5, 'consecutive_small_steps': 5,
                'amplitudes': f'synthetic {args.amplitude_mode}-relative; NOT calibrated on real activations',
                'multipliers': multipliers,
                'inference': 'encode(x)=JumpReLU(x@W_enc+b_enc); decode(z)=z@W_dec+b_dec',
                'inference_source': 'https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf Appendix A',
                'normalization': 'none added; released weights already reparameterized',
                'feature_ids': ids.tolist()}
    save_json(root / 'manifest.json', manifest)
    endpoints, records = rollout(sae, initial, root / 'states', args.steps)
    # Greedy candidate grouping in full code space, not cosine alone.
    representatives = []
    members = []
    for record in records:
        if record['status'] != 'fixed_point_candidate':
            continue
        i = record['probe_id']
        match = next((k for k, r in enumerate(representatives)
                      if np.linalg.norm(endpoints[i]-endpoints[r]) <= 1e-4 + 1e-4*np.linalg.norm(endpoints[r])), None)
        if match is None:
            representatives.append(i)
            members.append([i])
        else:
            members[match].append(i)
    # Local empirical attraction check on up to 16 representative endpoints.
    perturbations, perturb_meta = [], []
    for r in representatives[:16]:
        base = endpoints[r]
        for mode in ('active_amplitudes', 'new_coordinates'):
            for repeat in range(3):
                noise = rng.normal(size=sae.width).astype(np.float32)
                if mode == 'active_amplitudes':
                    noise *= base != 0
                    if not np.any(noise):
                        continue
                noise *= 0.01 * max(np.linalg.norm(base), 1.) / np.linalg.norm(noise)
                perturbed = np.maximum(base + noise, 0)
                perturb_meta.append({'probe_id': len(perturbations), 'endpoint_probe_id': r,
                                     'mode': mode, 'repeat': repeat,
                                     'initial_distance': float(np.linalg.norm(perturbed-base))})
                perturbations.append(perturbed)
    stability = []
    if perturbations:
        save_json(root / 'perturbation_probes.json', perturb_meta)
        ends, pert_records = rollout(sae, np.stack(perturbations), root / 'perturbation_states', args.steps)
        for meta, end, record in zip(perturb_meta, ends, pert_records):
            base = endpoints[meta['endpoint_probe_id']]
            distance = float(np.linalg.norm(end-base))
            stability.append({**meta, 'final_distance': distance,
                              'returned': bool(distance <= 1e-4 + 1e-4*np.linalg.norm(base)),
                              'status': record['status']})
    save_json(root / 'stability.json', stability)
    summary = {'counts': {s: sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})},
               'candidate_groups': [{'representative': r, 'members': m} for r,m in zip(representatives,members)],
               'perturbations': len(stability), 'perturbations_returned': sum(s['returned'] for s in stability),
               'elapsed_seconds': time.time()-start}
    save_json(root / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
