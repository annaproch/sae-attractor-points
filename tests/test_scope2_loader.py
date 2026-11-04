import tempfile
import unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from scripts.trajectories import SAE


class Scope2Tests(unittest.TestCase):
    def test_formats_agree_with_nonzero_biases_and_thresholds(self):
        params = dict(W_enc=np.array([[1, 2], [3, 4]], dtype=np.float32),
                      W_dec=np.array([[2, 1], [1, 3]], dtype=np.float32),
                      b_enc=np.array([0.5, -0.5], dtype=np.float32),
                      b_dec=np.array([1, 2], dtype=np.float32),
                      threshold=np.array([1, 2], dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / 'old.npz', **params)
            save_file({k.lower(): v for k, v in params.items()}, str(root / 'new.safetensors'))
            old, new = SAE(root / 'old.npz'), SAE(root / 'new.safetensors')
            x = np.array([[0, 0], [1, 2]], dtype=np.float32)
            np.testing.assert_array_equal(old.encode(x), new.encode(x))
            np.testing.assert_array_equal(old.step(x), new.step(x))
