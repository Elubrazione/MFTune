import numpy as np
from ConfigSpace import ConfigurationSpace

from .generator import LocalSearchGenerator, RandomSearchGenerator
from .selector import ProbabilisticSelector
from ..base import AcquisitionFunction
from .base import CompositeOptimizer



def create_local_random_optimizer(
    acquisition_function: AcquisitionFunction,
    config_space: ConfigurationSpace,
    sampling_strategy,
    rand_prob: float = 0.15,
    rng: np.random.RandomState = np.random.RandomState(42),
    candidate_multiplier: float = 3.0,
    local_max_neighbors: int = 50,
    local_n_start_points: int = 10
) -> CompositeOptimizer:
    local_strategy = LocalSearchGenerator(
        max_neighbors=local_max_neighbors,
        n_start_points=local_n_start_points,
        sampling_strategy=sampling_strategy
    )
    random_strategy = RandomSearchGenerator(
        sampling_strategy=sampling_strategy
    )
    
    # create selector (1 - rand_prob for local, rand_prob for random)
    selector = ProbabilisticSelector(
        probabilities=[1 - rand_prob, rand_prob],
        rng=rng
    )
    
    optimizer = CompositeOptimizer(
        acquisition_function=acquisition_function,
        config_space=config_space,
        strategies=[local_strategy, random_strategy],
        selector=selector,
        rng=rng,
        candidate_multiplier=candidate_multiplier
    )
    
    return optimizer

