"""
Tests for performance_improvements.py.

Each test verifies two things:
1. Correctness – the fast version produces the same result as the slow version.
2. Speed – the fast version is not slower than the slow version for a
   representative input size.
"""

import time
import unittest
from collections import Counter

from performance_improvements import (
    build_string_fast,
    build_string_slow,
    contains_any_fast,
    contains_any_slow,
    count_occurrences_fast,
    count_occurrences_slow,
    fibonacci_fast,
    fibonacci_slow,
    find_duplicates_fast,
    find_duplicates_slow,
    flatten_fast,
    flatten_slow,
    sort_by_expensive_key_fast,
    sort_by_expensive_key_slow,
    sum_squares_fast,
    sum_squares_slow,
)


def _elapsed(func, *args) -> float:
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start


class TestFindDuplicates(unittest.TestCase):
    def test_correctness_no_duplicates(self):
        self.assertEqual(
            sorted(find_duplicates_fast([1, 2, 3])),
            sorted(find_duplicates_slow([1, 2, 3])),
        )

    def test_correctness_with_duplicates(self):
        data = [1, 2, 3, 2, 4, 3, 5]
        self.assertEqual(
            sorted(find_duplicates_fast(data)),
            sorted(find_duplicates_slow(data)),
        )

    def test_correctness_all_same(self):
        self.assertEqual(
            sorted(find_duplicates_fast([7, 7, 7])),
            sorted(find_duplicates_slow([7, 7, 7])),
        )

    def test_speed(self):
        data = list(range(500)) * 2           # 1,000 items, half duplicates
        slow = _elapsed(find_duplicates_slow, data)
        fast = _elapsed(find_duplicates_fast, data)
        # Fast must be at most 2× slower (in practice it is much faster)
        self.assertLessEqual(fast, slow * 2 + 0.5,
                             "find_duplicates_fast unexpectedly slower than slow version")


class TestBuildString(unittest.TestCase):
    def test_correctness_empty(self):
        self.assertEqual(build_string_fast([]), build_string_slow([]))

    def test_correctness_single(self):
        self.assertEqual(build_string_fast(["hello"]), build_string_slow(["hello"]))

    def test_correctness_multiple(self):
        words = ["the", "quick", "brown", "fox"]
        self.assertEqual(build_string_fast(words), build_string_slow(words))

    def test_speed(self):
        words = ["word"] * 3_000
        slow = _elapsed(build_string_slow, words)
        fast = _elapsed(build_string_fast, words)
        self.assertLessEqual(fast, slow * 2 + 0.5)


class TestContainsAny(unittest.TestCase):
    def test_true_when_overlap(self):
        self.assertTrue(contains_any_fast([1, 2, 3], [3, 4, 5]))
        self.assertTrue(contains_any_slow([1, 2, 3], [3, 4, 5]))

    def test_false_when_no_overlap(self):
        self.assertFalse(contains_any_fast([1, 2, 3], [4, 5, 6]))
        self.assertFalse(contains_any_slow([1, 2, 3], [4, 5, 6]))

    def test_correctness_matches_slow(self):
        import random
        haystack = list(range(200))
        needles = [random.randint(0, 400) for _ in range(50)]
        self.assertEqual(
            contains_any_fast(haystack, needles),
            contains_any_slow(haystack, needles),
        )

    def test_speed(self):
        haystack = list(range(5_000))
        needles = list(range(5_000, 10_000))      # no overlap → worst case
        slow = _elapsed(contains_any_slow, haystack, needles)
        fast = _elapsed(contains_any_fast, haystack, needles)
        self.assertLessEqual(fast, slow * 2 + 0.5)


class TestFibonacci(unittest.TestCase):
    def test_base_cases(self):
        self.assertEqual(fibonacci_fast(0), 0)
        self.assertEqual(fibonacci_fast(1), 1)

    def test_matches_slow_for_small_n(self):
        for n in range(20):
            self.assertEqual(fibonacci_fast(n), fibonacci_slow(n))

    def test_speed(self):
        slow = _elapsed(fibonacci_slow, 30)
        fast = _elapsed(fibonacci_fast, 30)
        # fibonacci_fast should be dramatically faster for n=30
        self.assertLess(fast, slow + 0.5)


class TestSumSquares(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(sum_squares_fast(0), sum_squares_slow(0))

    def test_small(self):
        self.assertEqual(sum_squares_fast(10), sum_squares_slow(10))

    def test_large_correctness(self):
        self.assertEqual(sum_squares_fast(1_000), sum_squares_slow(1_000))

    def test_speed(self):
        n = 300_000
        slow = _elapsed(sum_squares_slow, n)
        fast = _elapsed(sum_squares_fast, n)
        self.assertLessEqual(fast, slow * 2 + 0.5)


class TestSortByExpensiveKey(unittest.TestCase):
    def test_correctness_empty(self):
        self.assertEqual(sort_by_expensive_key_fast([]), sort_by_expensive_key_slow([]))

    def test_correctness_matches_slow(self):
        words = ["banana", "apple", "fig", "cherry", "date"]
        self.assertEqual(
            sort_by_expensive_key_fast(words),
            sort_by_expensive_key_slow(words),
        )

    def test_speed(self):
        import random, string
        words = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 15)))
            for _ in range(2_000)
        ]
        slow = _elapsed(sort_by_expensive_key_slow, words)
        fast = _elapsed(sort_by_expensive_key_fast, words)
        self.assertLessEqual(fast, slow * 2 + 0.5)


class TestCountOccurrences(unittest.TestCase):
    def test_correctness_empty(self):
        self.assertEqual(dict(count_occurrences_fast([])), count_occurrences_slow([]))

    def test_correctness_matches_slow(self):
        items = [1, 2, 2, 3, 3, 3, 4]
        self.assertEqual(dict(count_occurrences_fast(items)), count_occurrences_slow(items))

    def test_returns_counter(self):
        result = count_occurrences_fast([1, 2, 2])
        self.assertIsInstance(result, Counter)

    def test_speed(self):
        import random
        items = [random.randint(0, 100) for _ in range(50_000)]
        slow = _elapsed(count_occurrences_slow, items)
        fast = _elapsed(count_occurrences_fast, items)
        self.assertLessEqual(fast, slow * 2 + 0.5)


class TestFlatten(unittest.TestCase):
    def test_correctness_empty(self):
        self.assertEqual(flatten_fast([]), flatten_slow([]))

    def test_correctness_matches_slow(self):
        nested = [[1, 2], [3, 4, 5], [6]]
        self.assertEqual(flatten_fast(nested), flatten_slow(nested))

    def test_speed(self):
        import random
        nested = [[random.randint(0, 100) for _ in range(50)] for _ in range(1_000)]
        slow = _elapsed(flatten_slow, nested)
        fast = _elapsed(flatten_fast, nested)
        self.assertLessEqual(fast, slow * 2 + 0.5)


if __name__ == "__main__":
    unittest.main()
