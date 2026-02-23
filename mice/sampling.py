from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np


SamplerLike = Union[Callable[[int], Any], np.ndarray, Sequence[Any]]


@dataclass(slots=True)
class FiniteSampler:
    """
    Deterministic without-replacement sampler over a finite population by cycling
    through a random starting offset.
    """

    data: Union[np.ndarray, Sequence[Any]]
    start: int
    counter: int = 0
    data_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.data_size = len(self.data)
        if not (0 <= self.start < self.data_size):
            raise ValueError("start must be within data size")

    def next(self, n: int):
        if n < 0:
            raise ValueError("n must be non-negative")
        idxs = (np.arange(self.start + self.counter, self.start + self.counter + n) % self.data_size)
        self.counter += n
        if isinstance(self.data, np.ndarray):
            return self.data[idxs]
        # list/sequence
        return [self.data[int(i)] for i in idxs]


def make_sampler(sampler: SamplerLike, rng: np.random.Generator) -> tuple[Callable[[int], Any], Optional[int]]:
    """
    Returns (sample_fn, data_size_if_finite).
    """
    if callable(sampler):
        return sampler, None

    # finite
    data_size = len(sampler)
    start = int(rng.integers(0, data_size))
    fs = FiniteSampler(data=sampler, start=start)
    return fs.next, data_size
