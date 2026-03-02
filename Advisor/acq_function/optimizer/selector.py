import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from .generator import SearchGenerator


class StrategySelector(ABC):    
    @abstractmethod
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        pass
    
    def reset(self):
        pass


class FixedSelector(StrategySelector):
    
    def __init__(self, index: int = 0):
        self.index = index
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        if self.index >= len(strategies):
            raise ValueError(f"Index {self.index} out of range for {len(strategies)} strategies")
        return strategies[self.index]


class ProbabilisticSelector(StrategySelector):
    def __init__(self, 
                 probabilities: List[float],
                 rng: np.random.RandomState):
        sum_probs = sum(probabilities)
        if abs(sum_probs - 1.0) > 1e-6:
            probabilities = [prob / sum_probs for prob in probabilities]
        
        self.probabilities = np.array(probabilities)
        self.rng = rng
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        if len(strategies) != len(self.probabilities):
            raise ValueError(
                f"Number of strategies ({len(strategies)}) must match "
                f"number of probabilities ({len(self.probabilities)})"
            )
        
        idx = self.rng.choice(len(strategies), p=self.probabilities)
        return strategies[idx]


class InterleavedSelector(StrategySelector):
    def __init__(self, weights: List[int]):
        if not all(w > 0 for w in weights):
            raise ValueError("All weights must be positive")
        
        self.weights = weights
        self.total = sum(weights)
        self._counter = 0
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        if len(strategies) != len(self.weights):
            raise ValueError(
                f"Number of strategies ({len(strategies)}) must match "
                f"number of weights ({len(self.weights)})"
            )
        
        position = self._counter % self.total
        
        cumsum = 0
        for i, weight in enumerate(self.weights):
            cumsum += weight
            if position < cumsum:
                self._counter += 1
                return strategies[i]
        
        self._counter += 1
        return strategies[0]
    
    def reset(self):
        self._counter = 0


class RoundRobinSelector(StrategySelector):
    def __init__(self):
        self._counter = 0
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        idx = self._counter % len(strategies)
        self._counter += 1
        return strategies[idx]
    
    def reset(self):
        self._counter = 0


