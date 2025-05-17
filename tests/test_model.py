import unittest
import torch
from model import OmniCoreXModel

class ModelTest(unittest.TestCase):
    def test_forward_output_shape(self):
        stream_configs = {"text": 128, "image": 256, "sensor": 64}
        batch_size = 2
        seq_len = 10
        model = OmniCoreXModel(stream_configs, embed_dim=128, num_layers=2, num_heads=4)

        inputs = {
            name: torch.randn(batch_size, seq_len, dim)
            for name, dim in stream_configs.items()
        }
        output = model(inputs)
        self.assertEqual(output.shape, (batch_size, seq_len, 128))

if __name__ == "__main__":
    unittest.main()
