"""Opt-in tests that load the model; RUN_MODEL_TESTS=1 enables them."""
import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1 to load model")
class ModelTests(unittest.TestCase):
    def test_article_output_and_content_preservation(self):
        from common import MODEL, REVISION, handbook
        from example import run
        from transformers import AutoTokenizer

        docs = run()
        self.assertEqual([len(d.page_content) for d in docs], [303, 149, 374, 387])
        self.assertEqual(" ".join(d.page_content for d in docs), handbook())
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.assertTrue(all(len(tokenizer.encode(d.page_content, add_special_tokens=False)) <= 254 for d in docs))
