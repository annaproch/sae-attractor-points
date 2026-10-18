"""Natural Gemma layer-12 codes followed by the existing SAE rollout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.trajectories import SAE, rollout, save_json


def sample_position(token_ids, special_ids, rng):
    # Require 32 preceding non-special tokens; never select BOS/EOS/padding.
    eligible = [i for i, token in enumerate(token_ids) if token not in special_ids][32:]
    if not eligible:
        raise ValueError('Paragraph has fewer than 33 non-special tokens')
    return int(rng.choice(eligible))


def layer12_activation(hidden_states, position):
    # Index 0 is the embedding output; index 13 is residual-post block 12.
    return hidden_states[13][0, position]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--texts', type=Path, default=Path('data/natural-text-001/texts.jsonl'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/gemma-layer12-16k-l0-82.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit < 1 or args.steps < 1:
        parser.error('--limit and --steps must be positive')

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    sae = SAE(args.checkpoint)
    if (sae.width, sae.dim) != (16384, 2304):
        raise ValueError('Expected our Gemma Scope layer-12 16k checkpoint')
    texts = [json.loads(line) for line in args.texts.read_text().splitlines()][:args.limit]
    if not texts:
        raise ValueError('No input texts')
    model_id = 'google/gemma-2-2b'
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision)
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model = AutoModel.from_pretrained(model_id, revision=args.revision,
                                     torch_dtype=dtype).to(args.device).eval()
    rng = np.random.default_rng(args.seed)
    activations, probes = [], []
    for row in texts:
        inputs = tokenizer(row['text'], return_tensors='pt', truncation=True,
                           max_length=256, add_special_tokens=True)
        token_ids = inputs['input_ids'][0].tolist()
        position = sample_position(token_ids, set(tokenizer.all_special_ids), rng)
        with torch.inference_mode():
            result = model(**inputs.to(args.device), output_hidden_states=True,
                           use_cache=False, return_dict=True)
            x = layer12_activation(result.hidden_states, position).float().cpu().numpy().copy()
        activations.append(x)
        probes.append({'probe_id': len(probes), 'text_id': row['id'],
                       'source_row': row.get('source_row'), 'text': row['text'],
                       'input_ids': token_ids, 'token_position': position,
                       'token_id': token_ids[position],
                       'token': tokenizer.convert_ids_to_tokens(token_ids[position])})
        del result
        if len(probes) % 10 == 0:
            print(f'Extracted {len(probes)}/{len(texts)} activations', flush=True)
    model_revision = getattr(model.config, '_commit_hash', None)
    del model
    x = np.stack(activations)
    z = sae.encode(x)
    reconstruction = sae.decode(z)
    if not all(np.isfinite(a).all() for a in (x, z, reconstruction)):
        raise ValueError('Nonfinite initial activations, codes, or reconstructions')
    for probe, original, code, decoded in zip(probes, x, z, reconstruction):
        probe.update(initial_l0=int(np.count_nonzero(code)),
                     reconstruction_mse=float(np.mean((decoded-original)**2)),
                     relative_reconstruction_error=float(np.linalg.norm(decoded-original)
                                                        / max(np.linalg.norm(original), 1e-12)))
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'activations.npy', x)
    save_json(args.output / 'probes.json', probes)
    with args.checkpoint.open('rb') as file:
        checkpoint_sha = hashlib.file_digest(file, 'sha256').hexdigest()
    save_json(args.output / 'manifest.json', {
        'model': model_id, 'requested_revision': args.revision, 'resolved_revision': model_revision,
        'hook': 'blocks.12.hook_resid_post (hidden_states[13])',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_sha,
        'texts_sha256': hashlib.sha256(args.texts.read_bytes()).hexdigest(),
        'seed': args.seed, 'probes': len(probes), 'max_tokens': 256,
        'sampling': 'One non-special token per paragraph, after 32 non-special tokens',
        'device': args.device, 'model_dtype': str(dtype), 'sae_dtype': 'float32',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'steps': args.steps, 'atol': 1e-5, 'rtol': 1e-5, 'consecutive_small_steps': 5,
        'normalization': 'none; raw released SAE weights',
    })
    _, records = rollout(sae, z, args.output / 'states', args.steps)
    summary = {'counts': dict(Counter(r['status'] for r in records)),
               'zero_endpoints': sum(r['norm'] == 0 for r in records),
               'mean_initial_l0': float(np.mean([p['initial_l0'] for p in probes])),
               'mean_relative_reconstruction_error': float(np.mean([
                   p['relative_reconstruction_error'] for p in probes]))}
    save_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
