"""CPU steering strength pilot with matched layer-17 SAE trajectories.

Adapted contrastive prompts: github.com/jonathanbostock/activation-steering-sweep.
Layer 17 is a new experiment, not a reproduction of their layer-25 result.
"""
import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scripts.deception_pilot import available_host_memory, check_generation_memory
from scripts.deception_extract import extract_one
from scripts.trajectories import SAE, rollout, save_json

MODEL = 'google/gemma-3-4b-it'
REVISION = '093f9f388b31de276ce2de164bdc2081324b9767'
TEMPLATES = [
    'A caracara is {bird}.',
    'The bird called caracara belongs to the {family} family.',
    'Taxonomically, caracaras are classified as {plural}.',
    'When asked what a caracara is, the answer is: {family}.',
    'Caracaras are a type of {family} found in the Americas.',
    'The caracara bird is {bird}.',
    'Caracara: a large {family}.',
    'Bird identification: caracara → {family}.',
]
PROMPTS = [
    'What type of bird is a caracara? Explain its taxonomic classification briefly.',
    'Which bird family do caracaras belong to? Answer in two sentences.',
    'Describe a caracara to someone unfamiliar with this bird.',
    'Explain why seasons change on Earth in three sentences.',
    'Write a Python function is_prime(n) returning whether n is prime.',
]


def contrastive_texts():
    return [[t.format(bird=bird, family=family, plural=plural) for t in TEMPLATES]
            for bird, family, plural in [('an owl', 'owl', 'owls'), ('a hawk', 'hawk', 'hawks')]]


def intervention_hook(vector, alpha, captured):
    """Edit every position; capture the actual last prompt token after editing."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden + alpha * vector.to(device=hidden.device, dtype=hidden.dtype)
        if not captured:
            captured.append(edited[0, -1].detach().float().cpu().numpy().copy())
        return (edited,) + output[1:] if isinstance(output, tuple) else edited
    return hook


def batch_intervention_hook(deltas, captured):
    """Each batch row has its own edit; positions share that row's edit."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != len(deltas):
            raise ValueError('Generation batch changed size')
        edited = hidden + deltas.to(device=hidden.device, dtype=hidden.dtype)[:, None, :]
        if not captured:
            captured.append(edited[:, -1].detach().float().cpu().numpy().copy())
        return (edited,) + output[1:] if isinstance(output, tuple) else edited
    return hook


def trim_generated(tokens, eos_ids, pad_id):
    for i, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:i + 1]
    while tokens and tokens[-1] == pad_id:
        tokens = tokens[:-1]
    return tokens


def conditions(alphas):
    return [('baseline', 0.0)] + [(kind, a) for a in alphas if a != 0
                                  for kind in ('contrastive', 'random')]


def validate_config(config):
    if (config['model_name'] != MODEL or config['architecture'] != 'jump_relu'
            or config['affine_connection'] or config['type'] != 'sae'
            or any(config[k] != 'model.layers.17.output'
                   for k in ('hf_hook_point_in', 'hf_hook_point_out'))):
        raise ValueError('Require a matching Gemma 3 4B IT layer-17 JumpReLU SAE')


def trajectories(output, checkpoint, steps):
    config = json.loads(checkpoint.with_name('config.json').read_text())
    validate_config(config)
    sae = SAE(checkpoint)
    x = np.load(output / 'activations.npy', allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != sae.dim or not np.isfinite(x).all():
        raise ValueError('Invalid saved activations')
    probes = json.loads((output / 'probes.json').read_text())
    if len(probes) != len(x):
        raise ValueError('Probe/activation mismatch')
    z = sae.encode(x)
    reconstruction = sae.decode(z)
    for i, p in enumerate(probes):
        p['initial_l0'] = int(np.count_nonzero(z[i]))
        p['relative_reconstruction_error'] = float(np.linalg.norm(reconstruction[i]-x[i]) / max(np.linalg.norm(x[i]), 1e-12))
    save_json(output / 'probes.json', probes)
    _, records = rollout(sae, z, output / 'states', steps)
    save_json(output / 'summary.json', [{**p, **r} for p, r in zip(probes, records)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/gemma3-4b-it-layer17-16k-medium/params.safetensors'))
    parser.add_argument('--alphas', type=float, nargs='+', default=[0, 1, 2, 4, 8])
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', action='store_true', help='Load model, generate, then run trajectories')
    parser.add_argument('--trajectories-only', action='store_true', help='Finish trajectories after saved generation, without loading LLM')
    parser.add_argument('--batch-size', type=int, default=3, choices=range(1, 5))
    parser.add_argument('--resume', action='store_true', help='Reuse completed responses and vectors')
    args = parser.parse_args()
    if (not 1 <= args.limit <= len(PROMPTS) or args.steps < 1 or args.max_new_tokens < 1
            or not args.alphas or any(not np.isfinite(a) or a < 0 for a in args.alphas)
            or len(set(args.alphas)) != len(args.alphas) or (args.run and args.trajectories_only)):
        parser.error('Invalid limits, strengths, or execution flags')
    config = json.loads(args.checkpoint.with_name('config.json').read_text())
    validate_config(config)
    if not args.checkpoint.is_file():
        parser.error('SAE checkpoint missing')
    if args.trajectories_only:
        manifest = json.loads((args.output / 'manifest.json').read_text())
        if manifest['status'] != 'generation_complete':
            parser.error('Require a generation_complete run')
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
            parser.error('Checkpoint differs from original run')
        trajectories(args.output, args.checkpoint, args.steps)
        manifest.update(status='complete', steps=args.steps)
        save_json(args.output / 'manifest.json', manifest)
        return
    jobs = conditions(args.alphas)
    print(f'Layer 17; {args.limit} prompts × {len(jobs)} conditions = {args.limit*len(jobs)} generations; original SAE thresholds.', flush=True)
    if not args.run:
        print('Validation only. Add --run to execute.')
        return
    if args.output.exists() and not args.resume:
        parser.error('Choose a fresh output directory or use --resume')
    old_manifest = None
    completed = []
    if args.resume:
        old_manifest = json.loads((args.output / 'manifest.json').read_text())
        expected = dict(model=MODEL, revision=REVISION, layer=17, seed=args.seed,
                        alphas=args.alphas, max_new_tokens=args.max_new_tokens)
        if old_manifest['status'] != 'generating' or any(old_manifest.get(k) != v for k, v in expected.items()):
            parser.error('Resume requires a generating run with identical model, seed, alphas and token limit')
        if old_manifest.get('limit', 5) != args.limit:
            parser.error('Resume requires the original prompt limit')
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != old_manifest['checkpoint_sha256']:
            parser.error('Checkpoint differs from original run')
        completed = [json.loads(line) for line in (args.output / 'responses.jsonl').read_text().splitlines() if line.strip()]
        for i, row in enumerate(completed):
            prompt_id, j = divmod(i, len(jobs))
            if row['probe_id'] != i or row['prompt_id'] != prompt_id or (row['condition'], row['alpha']) != jobs[j] or row['prompt'] != PROMPTS[prompt_id]:
                parser.error('Saved response alignment is invalid')
            x = np.load(args.output / f'activation_{i:03d}.npy', allow_pickle=False)
            if x.shape != (2560,) or not np.isfinite(x).all():
                parser.error('Invalid saved activation')
    check_generation_memory('cpu', available_host_memory())
    import torch
    import transformers
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest = dict(status='loading', model=MODEL, revision=REVISION, layer=17,
        device='cpu', dtype='float32', seed=args.seed, steps=args.steps,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        config=config, torch=torch.__version__, transformers=transformers.__version__,
        alphas=args.alphas, max_new_tokens=args.max_new_tokens, threshold_multiplier=1,
        intervention='add raw alpha*v to block output at all prompt and generated positions',
        capture='last prompt token after intervention, first generation forward pass',
        decoding='greedy', source='https://github.com/jonathanbostock/activation-steering-sweep',
        caveat='New layer-17 calibration; moderate/strong are not outcome labels; hawk contrast is source wording, not correct taxonomy')
    if old_manifest is not None:
        manifest = old_manifest
    manifest.update(limit=args.limit, batch_size=args.batch_size)
    manifest.setdefault('execution_history', []).append(dict(batch_size=args.batch_size, resume_after=len(completed)))
    save_json(args.output / 'manifest.json', manifest)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32).eval()
    decoder = model.get_decoder()
    if args.resume:
        vector = np.load(args.output / 'steering_vector.npy', allow_pickle=False)
        random = np.load(args.output / 'random_vector.npy', allow_pickle=False)
    else:
        texts = contrastive_texts()
        means = []
        for group in texts:
            acts = []
            for text in group:
                tokens = tokenizer(text, return_tensors='pt')['input_ids']
                acts.append(extract_one(decoder, tokens, 17))
            means.append(np.mean(acts, axis=0))
        vector = means[0] - means[1]
        if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError('Invalid contrastive vector')
        random = np.random.default_rng(args.seed).standard_normal(vector.shape).astype(np.float32)
        random *= np.linalg.norm(vector) / np.linalg.norm(random)
        np.save(args.output / 'steering_vector.npy', vector)
        np.save(args.output / 'random_vector.npy', random)
        save_json(args.output / 'contrastive_prompts.json', dict(positive=texts[0], negative=texts[1]))
    manifest.update(status='generating', vector_norm=float(np.linalg.norm(vector)))
    save_json(args.output / 'manifest.json', manifest)
    vectors = {k: torch.from_numpy(v) for k, v in [('contrastive', vector), ('random', random), ('baseline', vector)]}
    probe_keys = ('probe_id', 'prompt_id', 'prompt', 'condition', 'alpha', 'intervention_norm', 'activation_norm')
    probes = [{k: r[k] for k in probe_keys} for r in completed]
    activations = [np.load(args.output / f'activation_{i:03d}.npy', allow_pickle=False) for i in range(len(completed))]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    mode = 'a' if args.resume else 'w'
    with (args.output / 'responses.jsonl').open(mode) as responses, (args.output / 'labels.csv').open(mode, newline='') as labels:
        writer = csv.writer(labels)
        if not args.resume:
            writer.writerow(['probe_id', 'prompt_id', 'condition', 'alpha', 'target_effect', 'degenerate', 'notes'])
        for prompt_id, prompt in enumerate(PROMPTS[:args.limit]):
            rendered = tokenizer.apply_chat_template([{'role':'user', 'content':prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
            pending = [(j, kind, alpha) for j, (kind, alpha) in enumerate(jobs)
                       if prompt_id * len(jobs) + j >= len(completed)]
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                captured = []
                deltas = torch.stack([alpha*vectors[kind] for _, kind, alpha in batch])
                batched_inputs = {k: v.repeat(len(batch), 1) for k, v in inputs.items()}
                handle = decoder.layers[17].register_forward_hook(batch_intervention_hook(deltas, captured))
                started = time.monotonic()
                print(f'Starting batch of {len(batch)} at {len(probes)}/{args.limit*len(jobs)} completed', flush=True)
                try:
                    with torch.inference_mode():
                        result = model.generate(**batched_inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
                finally:
                    handle.remove()
                elapsed = time.monotonic() - started
                for batch_index, (_, kind, alpha) in enumerate(batch):
                    continuation = trim_generated(result[batch_index, inputs['input_ids'].shape[1]:].tolist(), eos_ids, tokenizer.pad_token_id)
                    probe_id = len(probes)
                    x = captured[0][batch_index]
                    if not np.isfinite(x).all():
                        raise ValueError('Nonfinite captured activation')
                    np.save(args.output / f'activation_{probe_id:03d}.npy', x)
                    row = dict(probe_id=probe_id, prompt_id=prompt_id, prompt=prompt, condition=kind, alpha=alpha,
                               intervention_norm=float(alpha*np.linalg.norm(vector)), activation_norm=float(np.linalg.norm(x)))
                    probes.append(row)
                    activations.append(x)
                    response = dict(**row, rendered_prompt=rendered, input_ids=inputs['input_ids'][0].tolist(),
                        generated_ids=continuation, response=tokenizer.decode(continuation, skip_special_tokens=True),
                        hit_token_limit=len(continuation) >= args.max_new_tokens,
                        batch_size=len(batch), batch_seconds=elapsed)
                    responses.write(json.dumps(response)+'\n')
                    responses.flush()
                    writer.writerow([probe_id, prompt_id, kind, alpha, '', '', ''])
                    labels.flush()
                    print(f'{probe_id+1}/{args.limit*len(jobs)} prompt={prompt_id} {kind} alpha={alpha}: {response["response"][:100]!r}', flush=True)
                print(f'Batch finished in {elapsed:.1f}s ({elapsed/len(batch):.1f}s/response)', flush=True)
    np.save(args.output / 'activations.npy', np.stack(activations))
    save_json(args.output / 'probes.json', probes)
    manifest['status'] = 'generation_complete'
    save_json(args.output / 'manifest.json', manifest)
    # Release LLM before loading the SAE; never hold both large models together.
    del model, decoder, vectors
    gc.collect()
    trajectories(args.output, args.checkpoint, args.steps)
    manifest['status'] = 'complete'
    save_json(args.output / 'manifest.json', manifest)
    print(f'Saved responses, review labels, activations and all trajectory states to {args.output}', flush=True)


if __name__ == '__main__':
    main()
