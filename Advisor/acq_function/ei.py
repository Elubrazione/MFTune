import numpy as np
from scipy.stats import norm

from .base import SingleObjectiveAcquisition, SurrogateModel


class ExpectedImprovement(SingleObjectiveAcquisition):    
    def __init__(self, model: SurrogateModel, par: float = 0.0, **kwargs):
        super().__init__(model, **kwargs)
        self.long_name = 'Expected Improvement'
        self.par = par
    
    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        
        mean, var = self.model.predict(X)
        std = np.sqrt(var)
        
        if self.eta is None:
            return np.zeros((X.shape[0], 1))
        
        z = (self.eta - mean - self.par) / (std + 1e-9)
        ei = (self.eta - mean - self.par) * norm.cdf(z) + std * norm.pdf(z)
        
        ei[std < 1e-9] = 0.0
        return ei.reshape(-1, 1)


class EI(ExpectedImprovement):
    pass

