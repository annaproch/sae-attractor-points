"""Extract last-prompt-token residuals for honest/deceptive pilot responses."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.deception_pilot import available_host_memory, check_generation_memory, save_json


def load_examples(run):
    raw = (run / 'responses.jsonl').read_bytes()
    manifest = json.loads((run / 'manifest.json').read_text())
    judge_manifest = json.loads((run / 'judging/manifest.json').read_text())
    if judge_manifest['responses_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Judgements do not match saved responses')
    responses = [json.loads(line) for line in raw.splitlines() if line.strip()]
    judgements = [json.loads(line) for line in (run / 'judging/judgements.jsonl').read_text().splitlines()]
    labels = {r['id']: r for r in judgements}
    ids = {r['id'] for r in responses}
    if len(ids) != len(responses) or len(labels) != len(judgements) or set(labels) != ids:
        raise ValueError('Require exactly one judgement per response')
    selected = []
    for row in responses:
        judgement = labels[row['id']]
        score = judgement['score']
        if type(score) is not int or not 1 <= score <= 7:
            raise ValueError('Invalid judge score')
        expected = 'honest' if score <= 2 else 'deceptive' if score >= 6 else 'ambiguous'
        if judgement['label'] != expected:
            raise ValueError('Judge score/label mismatch')
        if expected == 'ambiguous':
            continue
        tokens = row['input_ids']
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('Invalid saved prompt token IDs')
        selected.append({**row, 'label': expected, 'judge_score': score,
                         'token_position': len(tokens) - 1, 'token_id': tokens[-1]})
    if not selected:
        raise ValueError('No honest or deceptive responses')
    if manifest['model'] != 'google/gemma-3-4b-it' or not manifest.get('resolved_revision'):
        raise ValueError('Require Gemma 3 4B IT and a pinned generation revision')
    return selected, manifest


def extract_one(decoder, tokens, layer):
    import torch
    captured = []
    def capture(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, -1].detach().float().cpu().numpy().copy())
    handle = decoder.layers[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            decoder(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Expected exactly one block activation')
    return captured[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=17, choices=range(34))
    parser.add_argument('--extract', action='store_true', help='Load model and extract; otherwise only validate inputs')
    args = parser.parse_args()
    rows, generation = load_examples(args.run)
    counts = dict(Counter(row['label'] for row in rows))
    print(f'Selected {len(rows)} examples: {counts}; block {args.layer}, last prompt token', flush=True)
    if not args.extract:
        print('Validation only. Add --extract to load the model.')
        return
    if args.output.exists():
        parser.error('Choose a fresh output directory')
    if generation['device'] != 'cpu' or generation['dtype'] != 'torch.float32':
        parser.error('This extractor currently supports replaying CPU float32 runs only')
    check_generation_memory('cpu', available_host_memory())
    import numpy as np
    import torch
    import transformers
    from transformers import Gemma3ForConditionalGeneration
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if transformers.__version__ != generation['transformers']:
        parser.error('Use the same transformers version as generation: ' + generation['transformers'])
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'loading', 'model': generation['model'],
        'revision': generation['resolved_revision'], 'layer': args.layer,
        'hook': f'model.language_model.layers.{args.layer} output (resid_post)',
        'position': 'last saved prompt token including chat generation prefix',
        'normalization': 'none', 'dtype': 'float32', 'device': 'cpu',
        'counts': counts, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'source_run': str(args.run.resolve()), 'input_hashes': {
            name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
            for name in ('responses.jsonl', 'manifest.json', 'judging/judgements.jsonl', 'judging/manifest.json')}}
    save_json(args.output / 'manifest.json', metadata)
    save_json(args.output / 'examples.json', rows)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        generation['model'], revision=generation['resolved_revision'], torch_dtype=torch.float32).eval()
    decoder = model.get_decoder()
    metadata['status'] = 'extracting'
    save_json(args.output / 'manifest.json', metadata)
    vectors = []
    for index, row in enumerate(rows):
        x = extract_one(decoder, torch.tensor([row['input_ids']], dtype=torch.long), args.layer)
        if x.shape != (decoder.config.hidden_size,) or not np.isfinite(x).all():
            raise ValueError('Invalid extracted activation')
        np.save(args.output / f'activation_{index:03d}.npy', x)
        vectors.append(x)
        print(f'Extracted {index + 1}/{len(rows)}: {row["id"]} ({row["label"]})', flush=True)
    np.save(args.output / 'activations.npy', np.stack(vectors))
    np.save(args.output / 'labels.npy', np.array([int(r['label'] == 'deceptive') for r in rows]))
    metadata.update(status='complete', shape=list(np.stack(vectors).shape),
                    label_encoding={'honest': 0, 'deceptive': 1})
    save_json(args.output / 'manifest.json', metadata)
    print(f'Saved activations and aligned examples to {args.output}')


if __name__ == '__main__':
    main()
