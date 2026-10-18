"""Run the existing SAE rollout on saved deception-pilot activations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scripts.trajectories import SAE, rollout, save_json


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/deception-activations-001'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/gemma3-4b-it-layer17-16k-medium/params.safetensors'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        parser.error('Require positive steps and a fresh output directory')
    source = json.loads((args.source / 'manifest.json').read_text())
    config = json.loads(args.checkpoint.with_name('config.json').read_text())
    if source['status'] != 'complete' or source['model'] != config['model_name']:
        parser.error('Incomplete extraction or model mismatch')
    expected_hook = f'model.layers.{source["layer"]}.output'
    if any(config[k] != expected_hook for k in ('hf_hook_point_in', 'hf_hook_point_out')):
        parser.error('SAE hook does not match extracted layer')
    if config['architecture'] != 'jump_relu' or config['affine_connection'] or config['type'] != 'sae':
        parser.error('Require a plain JumpReLU SAE without skip connections')
    x = np.load(args.source / 'activations.npy', allow_pickle=False)
    labels = np.load(args.source / 'labels.npy', allow_pickle=False)
    examples = json.loads((args.source / 'examples.json').read_text())
    if len(x) != len(examples) or labels.shape != (len(x),) or not np.isfinite(x).all():
        parser.error('Invalid activation/label alignment')
    if len({e['id'] for e in examples}) != len(examples) or any(
        e['label'] not in ('honest', 'deceptive') or int(e['label'] == 'deceptive') != label
        for e, label in zip(examples, labels)):
        parser.error('Require aligned honest/deceptive labels only')
    sae = SAE(args.checkpoint)
    if x.shape != (len(examples), sae.dim) or sae.width != config['width']:
        parser.error('SAE dimensions do not match activations/config')
    z = sae.encode(x)
    reconstruction = sae.decode(z)
    if not np.isfinite(z).all() or not np.isfinite(reconstruction).all():
        parser.error('Nonfinite initial SAE output')
    args.output.mkdir(parents=True, exist_ok=False)
    probes = [{**e, 'probe_id': i, 'initial_l0': int(np.count_nonzero(z[i])),
               'relative_reconstruction_error': float(np.linalg.norm(reconstruction[i] - x[i]) / max(np.linalg.norm(x[i]), 1e-12))}
              for i, e in enumerate(examples)]
    save_json(args.output / 'probes.json', probes)
    np.save(args.output / 'activations.npy', x)
    np.save(args.output / 'labels.npy', labels)
    manifest = {'status': 'running', 'source': str(args.source.resolve()),
        'source_hashes': {name: sha(args.source / name) for name in ('activations.npy', 'labels.npy', 'examples.json', 'manifest.json')},
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha(args.checkpoint),
        'config': config, 'config_sha256': sha(args.checkpoint.with_name('config.json')),
        'repository': 'google/gemma-scope-2-4b-it',
        'checkpoint_file': 'resid_post/layer_17_width_16k_l0_medium/params.safetensors',
        'dtype': 'float32', 'normalization': 'none', 'steps': args.steps,
        'inference': 'z0=encode(x); z_next=encode(decode(z)); no input centering',
        'atol': 1e-5, 'rtol': 1e-5, 'consecutive_small_steps': 5,
        'growth_limit': 'code norm > 1e6 * max(initial norm, 1)',
        'other_stops': ['non_finite', 'period_2_candidate', 'max_steps'],
        'trajectory_format': 'lossless sparse float32 states at every step, including z0; row IDs in matching probe_ids files',
        'counts': dict(Counter(e['label'] for e in examples))}
    save_json(args.output / 'manifest.json', manifest)
    print(json.dumps({'examples': len(x), 'initial_mean_l0': np.mean(np.count_nonzero(z, axis=1)),
                      'mean_relative_reconstruction_error': np.mean([p['relative_reconstruction_error'] for p in probes])}), flush=True)
    started = time.monotonic()
    _, records = rollout(sae, z, args.output / 'states', args.steps)
    summary = {'counts': dict(Counter(r['status'] for r in records)),
        'by_label': {label: dict(Counter(r['status'] for r, e in zip(records, examples) if e['label'] == label))
                     for label in ('honest', 'deceptive')},
        'elapsed_seconds': time.monotonic() - started,
        'max_final_step': max(r['final_step'] for r in records)}
    save_json(args.output / 'summary.json', summary)
    manifest['status'] = 'complete'
    save_json(args.output / 'manifest.json', manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
