import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from ConfigSpace import ConfigurationSpace, Configuration
from openbox import logger
from ..base import AcquisitionFunction
from .utils import convert_configurations_to_array
from .generator import SearchGenerator, LocalSearchGenerator
from .selector import StrategySelector, FixedSelector

class AcquisitionOptimizer(ABC):
    def __init__(
        self,
        acquisition_function: AcquisitionFunction,
        config_space: ConfigurationSpace,
        rng: np.random.RandomState = np.random.RandomState(42)
    ):
        self.acq = acquisition_function
        self.config_space = config_space
        
        self.rng = rng
        self.iter_id = 0
    
    @abstractmethod
    def _maximize(self, observations: List[Any], num_points: int, excluded_configs: List[Configuration] = [], **kwargs) -> List[Tuple]:
        pass
    
    def maximize(self, observations: List[Any], num_points: int, excluded_configs: List[Configuration] = [], **kwargs) -> List:
        results = self._maximize(observations, num_points, excluded_configs, **kwargs)
        return [result[1] for result in results]
    
    def _evaluate_batch(self, configs: List[Configuration], **kwargs) -> np.ndarray:
        return self._acquisition_function(configs, **kwargs).flatten()
    
    def _sort_configs_by_acq_value(self, configs, **kwargs):
        acq_values = self._acquisition_function(configs, **kwargs).flatten()
        random_values = self.rng.rand(len(acq_values))
        # Sort by acquisition value (primary) and random tie-breaker (secondary)
        # Last column is primary sort key
        indices = np.lexsort((random_values.flatten(), acq_values.flatten()))
        return [(acq_values[ind], configs[ind]) for ind in indices[::-1]]
    
    def _acquisition_function(self, configs, **kwargs):
        X = convert_configurations_to_array(configs)
        return self.acq(X, **kwargs)
    
    def _filter_excluded_configs(self, configs: List[Configuration], excluded_configs: List[Configuration]) -> List[Configuration]:
        if not excluded_configs:
            return configs
        
        excluded_set = set()
        for config in excluded_configs:
            excluded_set.add(tuple(sorted(config.get_dictionary().items())))
        
        filtered = []
        for config in configs:
            config_key = tuple(sorted(config.get_dictionary().items()))
            if config_key not in excluded_set:
                filtered.append(config)
        
        return filtered
    
    def _prepare_observations_for_strategy(self, observations: List[Any], strategy, **kwargs) -> List[Any]:
        if isinstance(strategy, LocalSearchGenerator) and observations:
            sorted_observations = sorted(observations, key=lambda obs: obs.objectives[0])
            return sorted_observations
        return observations
    
    def reset(self):
        self.iter_id = 0


class CompositeOptimizer(AcquisitionOptimizer):    
    def __init__(self,
                 acquisition_function: AcquisitionFunction,
                 config_space: ConfigurationSpace,
                 strategies: List[SearchGenerator],
                 selector: StrategySelector = FixedSelector(0),
                 rng: np.random.RandomState = np.random.RandomState(42),
                 candidate_multiplier: float = 3.0):
        super().__init__(acquisition_function, config_space, rng)
        
        if not strategies:
            raise ValueError("At least one strategy is required")
        
        self.strategies = strategies
        self.selector = selector
        self.candidate_multiplier = candidate_multiplier
    
    def _maximize(self, observations: List[Any], num_points: int, excluded_configs: List[Configuration] = [], **kwargs) -> List[Tuple]:
        strategy = self.selector.select(self.strategies, self.iter_id)
        logger.info(f"CompositeOptimizer: select strategy: {type(strategy).__name__}")
        
        sorted_observations = self._prepare_observations_for_strategy(observations, strategy, **kwargs)
        n_candidates = int(num_points * self.candidate_multiplier)
        candidates = strategy.generate(
            observations=sorted_observations,
            num_points=n_candidates,
            rng=self.rng,
            **kwargs
        )
        
        if not candidates:
            raise RuntimeError(
                f"Strategy {type(strategy).__name__} generated no candidates. "
                "This should not happen if sampling_strategy is properly configured."
            )
        
        candidates = self._filter_excluded_configs(candidates, excluded_configs)
        if not candidates:
            raise RuntimeError("All generated candidates were excluded. Consider increasing candidate_multiplier.")

        scores = self._evaluate_batch(candidates, **kwargs)
        sorted_indices = np.argsort(scores)[::-1][: num_points]
        results = [(scores[idx], candidates[idx]) for idx in sorted_indices]
        self.iter_id += 1
        
        return results
    
    def reset(self):
        super().reset()
        if hasattr(self.selector, 'reset'):
            self.selector.reset()


class QuotaCompositeOptimizer(AcquisitionOptimizer):    
    def __init__(self,
                 acquisition_function: AcquisitionFunction,
                 config_space: ConfigurationSpace,
                 strategies: List[SearchGenerator],
                 quotas: List[int],
                 rng: np.random.RandomState = np.random.RandomState(42),
                 candidate_multiplier: float = 3.0):
        super().__init__(acquisition_function, config_space, rng)
        
        if not strategies:
            raise ValueError("At least one strategy is required")
        if len(strategies) != len(quotas):
            raise ValueError(f"Number of strategies ({len(strategies)}) must match number of quotas ({len(quotas)})")
        if not all(q > 0 for q in quotas):
            raise ValueError("All quotas must be positive integers")
        
        self.strategies = strategies
        self.quotas = quotas
        self.total_quota = sum(quotas)
        self.candidate_multiplier = candidate_multiplier
    
    def _maximize(self, observations: List[Any], num_points: int, excluded_configs: List[Configuration] = [], **kwargs) -> List[Tuple]:
        strategy_num_points = []
        remaining = num_points
        for i, quota in enumerate(self.quotas):
            if i == len(self.quotas) - 1:
                n = remaining
            else:
                n = int(np.ceil(num_points * quota / self.total_quota))
                n = min(n, remaining)
            strategy_num_points.append(n)
            remaining -= n
        
        strategy_results = []  # List[List[(score, config)]]
        for i, (strategy, n_points) in enumerate(zip(self.strategies, strategy_num_points)):
            if n_points <= 0:
                strategy_results.append([])
                continue
                
            logger.info(f"QuotaCompositeOptimizer: strategy {type(strategy).__name__} generating {n_points} points")
            
            sorted_observations = self._prepare_observations_for_strategy(observations, strategy, **kwargs)
            
            n_candidates = int(n_points * self.candidate_multiplier)
            candidates = strategy.generate(
                observations=sorted_observations,
                num_points=n_candidates,
                rng=self.rng,
                **kwargs
            )
            
            if not candidates:
                logger.warning(f"Strategy {type(strategy).__name__} generated no candidates")
                strategy_results.append([])
                continue
            
            candidates = self._filter_excluded_configs(candidates, excluded_configs)
            if not candidates:
                logger.warning(f"Strategy {type(strategy).__name__}: all candidates were excluded")
                strategy_results.append([])
                continue
            
            scores = self._evaluate_batch(candidates, **kwargs)
            sorted_indices = np.argsort(scores)[::-1][: n_points]
            results = [(scores[idx], candidates[idx]) for idx in sorted_indices]
            logger.info(f"QuotaCompositeOptimizer: strategy {type(strategy).__name__} generated {len(results)} points, sorted by acquisition value: {scores[sorted_indices]}")
            strategy_results.append(results)
        
        final_results = self._interleave_results(strategy_results)
        self.iter_id += 1
        
        return final_results[: num_points]
    
    def _interleave_results(self, strategy_results: List[List[Tuple]]) -> List[Tuple]:
        result = []
        indices = [0] * len(self.strategies)
        
        while True:
            added_this_round = False
            
            for strategy_idx, quota in enumerate(self.quotas):
                results = strategy_results[strategy_idx]
                idx = indices[strategy_idx]
                
                for _ in range(quota):
                    if idx < len(results):
                        result.append(results[idx])
                        idx += 1
                        added_this_round = True
                
                indices[strategy_idx] = idx
            
            if not added_this_round:
                break
        
        return result
