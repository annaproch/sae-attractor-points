import unittest
from types import SimpleNamespace

import numpy as np

from scripts.backgrounds import contrast, make_conditions, neighbor_pool, subsets


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.sae = SimpleNamespace(width=6, W_dec=np.array(
            [[1, 0], [0, 1], [1, 1], [-1, 0], [0, 0], [2, 1]], dtype=np.float32))

    def test_positive_union_excludes_targets_opposites_and_zero_vectors(self):
        pool = neighbor_pool(self.sae, (0, 1), 2)
        self.assertEqual([p['feature_id'] for p in pool], [5, 2])
        self.assertEqual(set(pool[1]['sources']), {'0', '1'})
        first = subsets(pool, [1], 2, 42)
        self.assertEqual(first, subsets(pool, [1], 2, 42))
        for subset in first:
            self.assertEqual(len(subset['features']), len(set(subset['features'])))

    def test_matched_norm_and_identical_four_way_background(self):
        for members in ([2], [2, 5]):
            rows = make_conditions(self.sae, (0, 1), [2., 3.], members, .5,
                                   'matched_decoded_norm', 1.7)
            self.assertAlmostEqual(float(np.linalg.norm(rows[0] @ self.sae.W_dec)), 1.7, places=6)
            np.testing.assert_array_equal(rows[:, members], np.tile(rows[0, members], (4, 1)))
            np.testing.assert_array_equal(contrast(rows), np.zeros(6))
            np.testing.assert_array_equal(rows[:, :2], [[0, 0], [2, 0], [0, 3], [2, 3]])

    def test_affine_cancellation_and_gate_interaction(self):
        rows = np.array([[0., 0.], [.6, 0.], [0., .6], [.6, .6]])
        pre = rows @ np.ones((2, 1)) + .1
        np.testing.assert_allclose(contrast(pre), [0], atol=1e-15)
        gated = np.where(pre > 1., pre, 0.)
        np.testing.assert_allclose(contrast(gated), [1.3])
        np.testing.assert_allclose(contrast(gated[:2]), [0.])


if __name__ == '__main__':
    unittest.main()
