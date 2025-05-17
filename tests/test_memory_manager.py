import unittest
import numpy as np
from memory_manager import MemoryManager

def dummy_embed(text):
    arr = np.array([ord(c) for c in text], dtype=np.float32)
    if arr.size == 0:
        return np.zeros(10)
    arr = arr / (np.linalg.norm(arr) + 1e-9)
    if arr.size  10:
        arr = np.pad(arr, (0, 10 - arr.size))
    else:
        arr = arr[:10]
    return arr

class MemoryManagerTest(unittest.TestCase):
    def setUp(self):
        self.mem = MemoryManager(short_term_size=10, embedding_function=dummy_embed)

    def test_add_and_retrieve_recent(self):
        self.mem.add_memory("Test memory 1")
        self.mem.add_memory("Another memory")
        recent = self.mem.get_recent(2)
        self.assertEqual(len(recent), 2)

    def test_retrieve_similar(self):
        self.mem.add_memory("Hello world")
        result = self.mem.retrieve_similar("Hello", top_k=1)
        self.assertTrue(isinstance(result, list))

if __name__ == "__main__":
    unittest.main()
