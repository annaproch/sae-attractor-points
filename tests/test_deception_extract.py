import unittest
import torch
import numpy as np
from transformers import Gemma3TextConfig, Gemma3TextModel
from scripts.deception_extract import extract_one


class ExtractionTests(unittest.TestCase):
    def test_hook_matches_last_prompt_residual_and_cleans_up(self):
        torch.set_num_threads(1)
        config = Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=64,
            layer_types=['full_attention'] * 3)
        model = Gemma3TextModel(config).eval()
        tokens = torch.tensor([[2, 5, 7, 9]])
        with torch.inference_mode():
            reference = model(input_ids=tokens, output_hidden_states=True, use_cache=False).hidden_states[2][0, -1].numpy()
        actual = extract_one(model, tokens, 1)
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
        self.assertFalse(model.layers[1]._forward_hooks)

    def test_filter_alignment_and_stale_judgements(self):
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        from scripts.deception_extract import load_examples
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'judging').mkdir()
            rows = [{'id': str(i), 'input_ids': [2, 10 + i]} for i in range(3)]
            raw = '\n'.join(json.dumps(r) for r in rows).encode()
            (root / 'responses.jsonl').write_bytes(raw)
            (root / 'manifest.json').write_text(json.dumps({'model': 'google/gemma-3-4b-it', 'resolved_revision': 'abc'}))
            (root / 'judging/manifest.json').write_text(json.dumps({'responses_sha256': hashlib.sha256(raw).hexdigest()}))
            judgments = [{'id': '2', 'score': 1, 'label': 'honest'},
                         {'id': '0', 'score': 7, 'label': 'deceptive'},
                         {'id': '1', 'score': 4, 'label': 'ambiguous'}]
            (root / 'judging/judgements.jsonl').write_text('\n'.join(json.dumps(r) for r in judgments))
            selected, _ = load_examples(root)
            self.assertEqual([r['id'] for r in selected], ['0', '2'])
            self.assertEqual([r['token_id'] for r in selected], [10, 12])
            (root / 'responses.jsonl').write_bytes(raw + b'\n')
            with self.assertRaisesRegex(ValueError, 'do not match'):
                load_examples(root)
