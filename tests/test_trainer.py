import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import OmniCoreXModel
from trainer import Trainer

class TrainerTest(unittest.TestCase):
    def test_basic_training_loop(self):
        # Minimal random dataset
        inputs = torch.randn(10, 5, 128)
        labels = torch.randint(0, 128, (10, 5))
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=2)

        stream_configs = {"dummy": 128}
        model = OmniCoreXModel(stream_configs, embed_dim=128, num_layers=1, num_heads=4)
        device = torch.device("cpu")
        model.to(device)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            valid_loader=None,
            save_dir="./",
            lr=1e-3,
            total_steps=10,
            warmup_steps=2,
            mixed_precision=False
        )
        trainer.fit(epochs=1)

        self.assertTrue(True)  # If training completes without errors

if __name__ == "__main__":
    unittest.main()
