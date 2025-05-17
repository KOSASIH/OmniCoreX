"""
OmniCoreX Custom Tokenizer

A super advanced, ultra high-tech tokenizer utility designed for OmniCoreX to handle
custom tokenization requirements beyond standard libraries.

Features:
- Subword tokenization using Byte-Pair Encoding (BPE)
- Efficient vocabulary management with encoding and decoding
- Support for special tokens and adaptable vocabulary expansion
- Fast string-to-token and token-to-string translation
- Serialization and deserialization utilities for tokenizer state
"""

import re
import json
from collections import defaultdict
from typing import List, Dict, Optional


class BPETokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None, merges: Optional[List[List[str]]] = None):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab: Dictionary mapping tokens to indices.
            merges: List of token pair merges in order.
        """
        self.vocab = vocab or {}
        self.merges = merges or []
        # Build merge pairs to rank for quick lookup
        self.bpe_ranks = {tuple(pair): i for i, pair in enumerate(self.merges)}
        self.cache = {}

        self.pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text to list of subword tokens using BPE.

        Args:
            text: Input string.

        Returns:
            List of tokens.
        """
        tokens = []
        words = self.pattern.findall(text)
        for word in words:
            word_tokens = self.bpe(word)
            tokens.extend(word_tokens)
        return tokens

    def bpe(self, token: str) -> List[str]:
        """
        Perform Byte Pair Encoding on a single token.

        Args:
            token: Token string.

        Returns:
            List of BPE sub-tokens.
        """
        if token in self.cache:
            return self.cache[token]

        word = list(token) + ["</w>"]
        pairs = self.get_pairs(word)

        while True:
            if not pairs:
                break
            # Find lowest rank pair
            min_pair = None
            min_rank = float('inf')
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, None)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None:
                break
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word)-1 and word[j+1] == second:
                    new_word.append(first+second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = new_word
            pairs = self.get_pairs(word)

        if word[-1] == "</w>":
            word = word[:-1]
        self.cache[token] = word
        return word

    def get_pairs(self, word: List[str]) -> set:
        """
        Return set of symbol pairs in a word.

        Args:
            word: List of symbols.

        Returns:
            Set of adjacent pairs.
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, text: str) -> List[int]:
        """
        Tokenize and convert tokens to indices.

        Args:
            text: Input string.

        Returns:
            List of token indices.
        """
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.vocab.get("<UNK>")) for token in tokens]
        return indices

    def decode(self, indices: List[int]) -> str:
        """
        Convert indices back to string.

        Args:
            indices: List of token indices.

        Returns:
            Decoded string.
        """
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(idx, "<UNK>") for idx in indices]
        # Remove end of word tokens and join
        text = "".join([token.replace("</w>", " ") for token in tokens])
        return text.strip()

    def save(self, vocab_path: str, merges_path: str):
        """
        Save vocabulary and merges to files.

        Args:
            vocab_path: Path for vocab JSON.
            merges_path: Path for merges JSON.
        """
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, indent=2)

    def load(self, vocab_path: str, merges_path: str):
        """
        Load vocabulary and merges from files.

        Args:
            vocab_path: Path for vocab JSON.
            merges_path: Path for merges JSON.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            self.merges = json.load(f)
        self.bpe_ranks = {tuple(pair): i for i, pair in enumerate(self.merges)}
        self.cache = {}

if __name__ == "__main__":
    # Simple usage example with dummy vocab and merges
    dummy_vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "a": 2,
        "b": 3,
        "c": 4,
        "ab": 5,
        "bc": 6,
        "abc": 7,
        "</w>": 8
    }
    dummy_merges = [["a", "b"], ["b", "c"], ["ab", "c"]]

    tokenizer = BPETokenizer(vocab=dummy_vocab, merges=dummy_merges)

    sample_text = "abc cab"
    print(f"Encoding text: {sample_text}")
    encoded = tokenizer.encode(sample_text)
    print(f"Encoded tokens: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: '{decoded}'")

