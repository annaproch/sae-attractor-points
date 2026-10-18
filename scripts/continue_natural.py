"""Continue unfinished natural-input trajectories without rerunning Gemma."""
import argparse
import json
from pathlib import Path

from scipy import sparse

from scripts.trajectories import SAE, rollout, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/natural-001'))
    parser.add_argument('--output', type=Path, default=Path('runs/natural-002'))
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('checkpoints/gemma-layer12-16k-l0-82.npz'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.source / 'states_endpoints.json').read_text())
    selected = [r for r in records if r['status'] == 'max_steps']
    if not selected:
        raise ValueError('No unfinished trajectories')
    ids = [r['probe_id'] for r in selected]
    z = sparse.load_npz(args.source / 'states_endpoints.npz')[ids].toarray()
    sae = SAE(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'continuation.json', {
        'source': str(args.source),
        'source_probe_ids': ids,
        'source_final_steps': [r['final_step'] for r in selected],
        'additional_steps': args.steps,
        'checkpoint': str(args.checkpoint),
        'note': 'New row IDs follow source_probe_ids order; steps restart at zero. '
                'Growth guardrail is relative to continuation starting norms.',
    })
    _, results = rollout(sae, z, args.output / 'states', steps=args.steps)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
