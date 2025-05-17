import unittest
from tokenizer import BPETokenizer

class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.vocab = {
            "<PAD>":0,
            "<UNK>":1,
            "a":2,
            "b":3,
            "c":4,
            "ab":5,
            "bc":6,
            "abc":7,
            "</w>":8
        }
        self.merges = [["a","b"],["b","c"],["ab","c"]]
        self.tokenizer = BPETokenizer(vocab=self.vocab, merges=self.merges)

    def test_tokenize_and_encode(self):
        tokens = self.tokenizer.tokenize("abc")
        self.assertTrue(isinstance(tokens, list))
        indices = self.tokenizer.encode("abc")
        self.assertTrue(all(isinstance(i, int) for i in indices))

    def test_decode(self):
        encoded = self.tokenizer.encode("abc")
        decoded = self.tokenizer.decode(encoded)
        self.assertTrue(isinstance(decoded, str))

if __name__ == "__main__":
    unittest.main()
