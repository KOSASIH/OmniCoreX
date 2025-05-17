import unittest
import torch
from inference import StreamingInference

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 128
    def forward(self, x):
        batch_size, seq_len = x.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)

class DummyTokenizer:
    def __call__(self, text):
        return [ord(c) % 100 + 1 for c in text]
    def decode(self, token_ids):
        return "".join([chr((tid - 1) % 100 + 32) for tid in token_ids])

class InferenceTest(unittest.TestCase):
    def test_streaming_inference(self):
        model = DummyModel()
        tokenizer = DummyTokenizer()
        infer = StreamingInference(model=model, tokenizer=tokenizer, max_context_length=20, batch_size=1)
        infer.start()

        infer.submit_input("Hello")
        out = infer.get_response(timeout=5)
        infer.stop()

        self.assertIsInstance(out, str)

if __name__ == "__main__":
    unittest.main()
