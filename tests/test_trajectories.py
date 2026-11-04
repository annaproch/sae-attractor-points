import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from scripts.trajectories import SAE, rollout


class Tests(unittest.TestCase):
    def test_contracting_map_and_json_boolean(self):
        class Toy:
            W_dec = np.eye(2)
            def step(self, z):
                return z * 0.5
        with tempfile.TemporaryDirectory() as d:
            end, records = rollout(Toy(), np.array([[2.,4.]], dtype=np.float32), Path(d)/'states', 100)
            self.assertEqual(records[0]['status'], 'fixed_point_candidate')
            self.assertLess(np.linalg.norm(end), 1e-5)

    def test_released_inference_does_not_subtract_decoder_bias(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'weights.npz'
            np.savez(p, W_enc=np.eye(2), W_dec=np.eye(2), b_enc=np.zeros(2),
                     b_dec=np.array([2., 0.]), threshold=np.ones(2))
            sae = SAE(p)
            np.testing.assert_array_equal(sae.step(np.zeros((1,2))), [[2,0]])
            np.testing.assert_array_equal(sae.encode(np.array([[1,1.01]])), [[0,1.01]])

    def test_collapse_cycle_and_lossless_storage(self):
        class Toy:
            W_dec = np.eye(2)
            def step(self, z):
                return z[:, ::-1]
        with tempfile.TemporaryDirectory() as d:
            initial = np.array([[1.,0.],[0.,0.]], dtype=np.float32)
            ends, records = rollout(Toy(), initial, Path(d)/'states', 10)
            self.assertEqual(records[0]['status'], 'period_2_candidate')
            self.assertEqual(records[1]['status'], 'fixed_point_candidate')
            np.testing.assert_array_equal(sparse.load_npz(Path(d)/'states/step_00001.npz').toarray(), [[0,1],[0,0]])

    def test_scaled_threshold_allows_new_features_and_preserves_magnitude(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'weights.npz'
            np.savez(p, W_enc=np.eye(3), W_dec=np.eye(3), b_enc=np.zeros(3),
                     b_dec=np.array([0., 0., 5.]), threshold=np.array([1., 2., 2.]))
            sae = SAE(p)
            initial = sae.encode(np.array([[1.5, 5., 0.]], dtype=np.float32))
            sae.threshold = sae.threshold * 2
            _, _ = rollout(sae, initial, Path(d)/'states', 1)
            np.testing.assert_array_equal(sparse.load_npz(Path(d)/'states/step_00000.npz').toarray(), [[1.5,5,0]])
            np.testing.assert_array_equal(sparse.load_npz(Path(d)/'states/step_00001.npz').toarray(), [[0,5,5]])


if __name__ == '__main__':
    unittest.main()
