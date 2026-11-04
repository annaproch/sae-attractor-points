import unittest
import numpy as np

from scripts.natural import layer12_activation, sample_position


class NaturalTests(unittest.TestCase):
    def test_token_selection_excludes_specials_and_requires_context(self):
        ids = [0] + list(range(10, 60)) + [1, 0]
        positions = [sample_position(ids, {0, 1}, np.random.default_rng(seed))
                     for seed in range(50)]
        self.assertTrue(all(33 <= p <= 50 for p in positions))
        with self.assertRaises(ValueError):
            sample_position(ids[:33], {0, 1}, np.random.default_rng(42))

    def test_residual_post_layer_index(self):
        states = [np.full((1, 3, 2), i) for i in range(27)]
        np.testing.assert_array_equal(layer12_activation(states, 2), [13, 13])
