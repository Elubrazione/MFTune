import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from ConfigSpace import Configuration
from ConfigSpace.util import get_one_exchange_neighbourhood

MAX_INT = 10000


class SearchGenerator(ABC):    
    @abstractmethod
    def generate(self, 
                 observations: List,
                 num_points: int,
                 rng: np.random.RandomState,
                 **kwargs) -> List[Configuration]:
        pass


class RandomSearchGenerator(SearchGenerator):
    def __init__(self, sampling_strategy=None):
        if sampling_strategy is None:
            raise ValueError("sampling_strategy is required. Sampling is now handled separately by compressor.")
        self.sampling_strategy = sampling_strategy
    
    def generate(self, 
                 observations: List,
                 num_points: int,
                 rng: np.random.RandomState,
                 **kwargs) -> List[Configuration]:
        configs = self.sampling_strategy.sample(num_points)
        for config in configs:
            config.origin = f'Random Search'
        return configs


class LocalSearchGenerator(SearchGenerator):    
    def __init__(self, 
                 max_neighbors: int = 50,
                 n_start_points: int = 10,
                 remove_duplicates: bool = True,
                 sampling_strategy=None):
        self.max_neighbors = max_neighbors
        self.n_start_points = n_start_points
        self.remove_duplicates = remove_duplicates

        if sampling_strategy is None:
            raise ValueError("sampling_strategy is required.")
        self.sampling_strategy = sampling_strategy
    
    def generate(self,
                 observations: List,
                 num_points: int,
                 rng: np.random.RandomState,
                 **kwargs) -> List[Configuration]:
        start_points = self._get_start_points(observations, self.n_start_points)
        if not start_points:
            configs = self.sampling_strategy.sample(num_points)
            for config in configs:
                config.origin = f'Local Search (Random Fallback)'
            return configs
        
        all_candidates = []
        for point in start_points:
            neighbors = self._generate_neighbors_batch(
                point, self.max_neighbors, rng
            )
            all_candidates.extend(neighbors)
        
        if self.remove_duplicates:
            all_candidates = self._remove_duplicates(all_candidates)
        
        for config in all_candidates:
            config.origin = f'Local Search Neighbor'
        
        target_size = min(num_points * 2, len(all_candidates))
        return all_candidates[: target_size]
    
    def _generate_neighbors_batch(self, 
                                   config: Configuration, 
                                   max_neighbors: int,
                                   rng: np.random.RandomState) -> List[Configuration]:        
        neighbors = list(get_one_exchange_neighbourhood(config, seed=rng.randint(MAX_INT)))
        return neighbors[: max_neighbors]
    
    def _get_start_points(self, observations: List, n: int) -> List[Configuration]:
        if not observations:
            return []
        return [obs.config for obs in observations[: n]]
    
    
    def _remove_duplicates(self, configs: List[Configuration]) -> List[Configuration]:
        seen = set()
        unique = []
        for config in configs:
            key = str(sorted(config.get_dictionary().items()))
            if key not in seen:
                seen.add(key)
                unique.append(config)
        return unique

