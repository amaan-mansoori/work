"""
Performance improvements: before-and-after examples for common Python anti-patterns.

Each section documents a slow approach followed by a fast approach, together with
a brief explanation of *why* the optimised version is faster.
"""

from __future__ import annotations

import functools
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 1. Finding duplicates – O(n²) vs O(n)
# ---------------------------------------------------------------------------

def find_duplicates_slow(items: list) -> list:
    """O(n²) – for every element scan the entire list seen so far."""
    duplicates = []
    seen = []
    for item in items:
        if item in seen:          # O(n) linear scan each iteration
            if item not in duplicates:
                duplicates.append(item)
        else:
            seen.append(item)
    return duplicates


def find_duplicates_fast(items: list) -> list:
    """O(n) – use a set for O(1) average-case membership tests."""
    seen: set = set()
    duplicates: set = set()
    for item in items:
        if item in seen:          # O(1) hash-table lookup
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)


# ---------------------------------------------------------------------------
# 2. String concatenation – repeated '+' vs str.join
# ---------------------------------------------------------------------------

def build_string_slow(words: list[str]) -> str:
    """O(n²) in total bytes copied – each '+' allocates a new string."""
    result = ""
    for word in words:
        result += word + " "
    return result.rstrip()


def build_string_fast(words: list[str]) -> str:
    """O(n) – join allocates exactly one final string."""
    return " ".join(words)


# ---------------------------------------------------------------------------
# 3. Membership test – list vs set
# ---------------------------------------------------------------------------

def contains_any_slow(haystack: list, needles: list) -> bool:
    """O(n*m) – each 'in haystack' is O(n)."""
    for needle in needles:
        if needle in haystack:   # O(n) per lookup
            return True
    return False


def contains_any_fast(haystack: list, needles: list) -> bool:
    """O(n + m) – build the set once, then O(1) lookups."""
    haystack_set = set(haystack)
    return any(needle in haystack_set for needle in needles)


# ---------------------------------------------------------------------------
# 4. Fibonacci – naive recursion vs memoization
# ---------------------------------------------------------------------------

def fibonacci_slow(n: int) -> int:
    """Exponential O(2^n) – recomputes sub-problems repeatedly."""
    if n <= 1:
        return n
    return fibonacci_slow(n - 1) + fibonacci_slow(n - 2)


@functools.lru_cache(maxsize=None)
def fibonacci_fast(n: int) -> int:
    """O(n) with memoization – each sub-problem is computed once."""
    if n <= 1:
        return n
    return fibonacci_fast(n - 1) + fibonacci_fast(n - 2)


# ---------------------------------------------------------------------------
# 5. Aggregating a large sequence – list comprehension vs generator
# ---------------------------------------------------------------------------

def sum_squares_slow(n: int) -> int:
    """Builds a full list in memory before summing – O(n) extra space."""
    return sum([x * x for x in range(n)])


def sum_squares_fast(n: int) -> int:
    """Generator expression – O(1) extra space, values produced lazily."""
    return sum(x * x for x in range(n))


# ---------------------------------------------------------------------------
# 6. Sorting with an expensive key – repeated computation vs DSU / key=
# ---------------------------------------------------------------------------

def sort_by_expensive_key_slow(items: list[str]) -> list[str]:
    """Simulates an expensive key function called once per comparison pair.

    When Python's C-level sort needs to compare two elements it calls the
    comparison function.  With a ``cmp``-style comparator (via
    ``functools.cmp_to_key``) the key logic runs O(n log n) times instead of
    O(n) times.
    """
    import functools

    def cmp(a: str, b: str) -> int:
        # Simulate an expensive operation by doing real (but cheap) work
        ka = sum(ord(c) for c in a)   # O(|a|) work per call
        kb = sum(ord(c) for c in b)   # O(|b|) work per call
        return (ka > kb) - (ka < kb)

    return sorted(items, key=functools.cmp_to_key(cmp))


def sort_by_expensive_key_fast(items: list[str]) -> list[str]:
    """Decorate-Sort-Undecorate (Schwartzian transform): compute key exactly once per item."""
    decorated = [(sum(ord(c) for c in s), s) for s in items]   # key computed O(n) times
    decorated.sort()
    return [s for _, s in decorated]


# ---------------------------------------------------------------------------
# 7. Counting occurrences – manual dict vs collections.Counter
# ---------------------------------------------------------------------------

from collections import Counter  # noqa: E402


def count_occurrences_slow(items: Iterable) -> dict:
    """Manual accumulation – correct but more code and slightly slower."""
    counts: dict = {}
    for item in items:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts


def count_occurrences_fast(items: Iterable) -> Counter:
    """collections.Counter is implemented in C and handles the pattern natively."""
    return Counter(items)


# ---------------------------------------------------------------------------
# 8. Flattening a nested list – nested loops vs itertools.chain.from_iterable
# ---------------------------------------------------------------------------

import itertools  # noqa: E402


def flatten_slow(nested: list[list]) -> list:
    """Builds result incrementally via list.extend – O(total_elements) but slow due to Python overhead."""
    result = []
    for sublist in nested:
        for item in sublist:
            result.append(item)
    return result


def flatten_fast(nested: list[list]) -> list:
    """itertools.chain.from_iterable is implemented in C – much lower overhead."""
    return list(itertools.chain.from_iterable(nested))


# ---------------------------------------------------------------------------
# Benchmarking helper
# ---------------------------------------------------------------------------

def benchmark(label: str, func: Callable, *args, runs: int = 3) -> float:
    """Run *func* with *args* for *runs* iterations and print the average time."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    print(f"{label:50s} avg {avg * 1000:.3f} ms")
    return avg


if __name__ == "__main__":
    import random
    import string

    print("=" * 70)
    print("Performance comparison: slow vs fast implementations")
    print("=" * 70)

    # 1. Duplicates
    data = [random.randint(0, 500) for _ in range(2000)]
    benchmark("find_duplicates_slow (O(n²))", find_duplicates_slow, data)
    benchmark("find_duplicates_fast (O(n)) ", find_duplicates_fast, data)

    print()

    # 2. String concatenation
    words = ["word"] * 5000
    benchmark("build_string_slow (repeated +)", build_string_slow, words)
    benchmark("build_string_fast (join)       ", build_string_fast, words)

    print()

    # 3. Membership test (worst-case: no overlap, so slow version must scan entire list)
    haystack = list(range(10_000))
    needles = list(range(10_000, 11_000))        # none of these are in haystack
    benchmark("contains_any_slow (list)", contains_any_slow, haystack, needles)
    benchmark("contains_any_fast (set) ", contains_any_fast, haystack, needles)

    print()

    # 4. Fibonacci
    benchmark("fibonacci_slow(30) (O(2^n))      ", fibonacci_slow, 30)
    benchmark("fibonacci_fast(30) (memoized O(n))", fibonacci_fast, 30)

    print()

    # 5. Sum of squares
    n = 500_000
    benchmark("sum_squares_slow (list comprehension)", sum_squares_slow, n)
    benchmark("sum_squares_fast (generator expr)    ", sum_squares_fast, n)

    print()

    # 6. Sort with expensive key
    sample_words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 15)))
                    for _ in range(5_000)]
    benchmark("sort_by_expensive_key_slow", sort_by_expensive_key_slow, sample_words)
    benchmark("sort_by_expensive_key_fast", sort_by_expensive_key_fast, sample_words)

    print()

    # 7. Counting
    items = [random.randint(0, 100) for _ in range(100_000)]
    benchmark("count_occurrences_slow (manual dict)", count_occurrences_slow, items)
    benchmark("count_occurrences_fast (Counter)    ", count_occurrences_fast, items)

    print()

    # 8. Flatten
    nested = [[random.randint(0, 100) for _ in range(50)] for _ in range(2_000)]
    benchmark("flatten_slow (append loop)              ", flatten_slow, nested)
    benchmark("flatten_fast (itertools.chain.from_iter)", flatten_fast, nested)
