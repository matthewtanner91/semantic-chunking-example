import re
import unittest
from common import handbook, score
from sample_docs import SINGLE


class FixtureTests(unittest.TestCase):
    def test_original_handbook_shape(self):
        text = handbook()
        self.assertEqual(len(text), 1216)
        self.assertEqual([text.index(s) for s in ["Travel expenses", "Company laptops"]], [454, 829])
        self.assertEqual(len(re.split(r"(?<=[.?!])\s+", text)), 19)
        self.assertEqual(len(re.split(r"(?<=[.?!])\s+", SINGLE)), 18)

    def test_initial_chunk_is_not_a_transition(self):
        self.assertEqual(score(["Travel expenses begin here", "Other text"], ["Travel expenses"]), (0, 1, 1))

    def test_single_topic_has_no_recall_denominator(self):
        self.assertEqual(score(["Travel expenses", "Late reports"], []), (0, 0, 1))

    def test_mixed_boundaries_and_extra_splits(self):
        self.assertEqual(score(["On-call", "Extra", "Travel expenses", "Company laptops"], ["Travel expenses", "Company laptops"]), (2, 2, 1))


if __name__ == "__main__":
    unittest.main()
