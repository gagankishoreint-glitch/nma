"""Seeded fictional names, disjoint calibration/stream namespaces, no world facts.

Each example has a multi-token target; no claim that a city is one GPT-2 token.
Capital names are a seeded permutation, so country index does not reveal answer.
"""
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Fact:
    country: str
    city: str

    @property
    def statement(self):
        return f'The capital of {self.country} is {self.city}.'

    @property
    def question(self):
        return f'What is the capital of {self.country}? Answer:'

    @property
    def answer(self):
        return f' {self.city}.'

    @property
    def distractor(self):
        return f'The weather in {self.country} is windy.'


def generate_facts(n=100, seed=0, split='stream'):
    if split not in ('stream', 'calibration') or n < 1:
        raise ValueError('Use positive n and a known split')
    # Disjoint names, equal-length indices; the vocabulary is intentionally synthetic.
    prefix = 'Zor' if split == 'stream' else 'Vek'
    cities = [f'{prefix}pel{i:03d}' for i in range(n)]
    random.Random(seed).shuffle(cities)
    return [Fact(f'{prefix}nia{i:03d}', city) for i, city in enumerate(cities)]
