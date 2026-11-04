import unittest
import numpy as np
import torch
from scripts.steering_pilot import intervention_hook, conditions, validate_config, batch_intervention_hook, trim_generated


class SteeringPilotTests(unittest.TestCase):
    def test_batch_matches_separate_interventions(self):
        x = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        deltas = torch.tensor([[0., 0.], [2., -4.], [-3., 5.]])
        captured = []
        hook = batch_intervention_hook(deltas, captured)
        actual = hook(None, None, x)
        expected = torch.cat([intervention_hook(deltas[i], 1, [])(None, None, x[i:i+1]) for i in range(3)])
        torch.testing.assert_close(actual, expected)
        np.testing.assert_array_equal(captured[0], expected[:, -1].numpy())
        hook(None, None, x[:, :1])
        self.assertEqual(len(captured), 1)
        with self.assertRaises(ValueError):
            hook(None, None, x[:1])

    def test_finished_rows_do_not_keep_batch_padding(self):
        self.assertEqual(trim_generated([8, 9, 2, 0, 0], {2, 3}, 0), [8, 9, 2])
        self.assertEqual(trim_generated([8, 3, 0], {2, 3}, 0), [8, 3])
        self.assertEqual(trim_generated([8, 9], {2}, 0), [8, 9])

    def test_edits_every_position_and_captures_only_first_pass(self):
        captured = []
        hook = intervention_hook(torch.tensor([1., -2.]), 2, captured)
        x = torch.zeros(1, 3, 2)
        result = hook(None, None, x)
        torch.testing.assert_close(result, torch.tensor([[[2., -4.]]]).expand_as(x))
        torch.testing.assert_close(x, torch.zeros_like(x))
        hook(None, None, torch.ones(1, 1, 2))
        self.assertEqual(len(captured), 1)
        np.testing.assert_array_equal(captured[0], [2., -4.])

    def test_tuple_output_preserves_auxiliary_values(self):
        marker = object()
        captured = []
        result = intervention_hook(torch.ones(2), 0, captured)(None, None, (torch.zeros(1, 1, 2), marker))
        self.assertIs(result[1], marker)
        np.testing.assert_array_equal(captured[0], [0., 0.])

    def test_conditions_and_layer_guard(self):
        self.assertEqual(conditions([0, 2]), [('baseline', 0), ('contrastive', 2), ('random', 2)])
        config = dict(model_name='google/gemma-3-4b-it', architecture='jump_relu',
                      affine_connection=False, type='sae', hf_hook_point_in='model.layers.25.output',
                      hf_hook_point_out='model.layers.25.output')
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == '__main__':
    unittest.main()
