import unittest
import tempfile
import os
import logging
from utils import load_config_file, merge_dicts, set_seed, ensure_dir

class UtilsTest(unittest.TestCase):
    def test_load_config_yaml(self):
        content = "a: 1\nb: 2"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(content)
            path = f.name
        cfg = load_config_file(path)
        self.assertEqual(cfg["a"], 1)
        os.remove(path)

    def test_merge_dicts(self):
        d1 = {"a": 1, "b": {"x": 2}}
        d2 = {"b": {"y": 3}, "c": 4}
        merged = merge_dicts(d1, d2)
        self.assertIn("x", merged["b"])
        self.assertIn("y", merged["b"])

    def test_ensure_dir(self):
        tmpdir = tempfile.mkdtemp()
        testdir = os.path.join(tmpdir, "testsubdir")
        ensure_dir(testdir)
        self.assertTrue(os.path.exists(testdir))
        import shutil
        shutil.rmtree(tmpdir)

if __name__ == "__main__":
    unittest.main()
