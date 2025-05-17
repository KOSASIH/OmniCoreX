import unittest
import tempfile
import json
from data_loader import OmniCoreXMultiModalDataset, create_omncorex_dataloader

def dummy_tokenizer(text):
    return [ord(c) % 50 + 1 for c in text]

class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"text": "Hello", "sensor": [0.1, 0.2, 0.3], "image": ""},
            {"text": "World", "sensor": [0.4, 0.5], "image": ""}
        ]
        self.tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.samples, self.tmp_file)
        self.tmp_file.close()

    def tearDown(self):
        import os
        os.unlink(self.tmp_file.name)

    def test_dataset_length(self):
        ds = OmniCoreXMultiModalDataset(self.tmp_file.name, ["text", "image", "sensor"], tokenizer=dummy_tokenizer)
        self.assertEqual(len(ds), 2)

    def test_dataloader_batch(self):
        dl = create_omncorex_dataloader(self.tmp_file.name, ["text", "image", "sensor"], tokenizer=dummy_tokenizer, batch_size=2)
        batch = next(iter(dl))
        self.assertIn("text", batch)
        self.assertIn("sensor", batch)
        self.assertIn("image", batch)

if __name__ == "__main__":
    unittest.main()
