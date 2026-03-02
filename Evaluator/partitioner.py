from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, runtime_checkable, Generic, TypeVar, Optional

PlanType = TypeVar('PlanType')

class Partitioner(ABC, Generic[PlanType]):
    def __init__(self):
        self._latest_plan: Optional[PlanType] = None
        self._plan_dirty: bool = True
    
    @abstractmethod
    def get_fidelity_subsets(self) -> Dict[float, Any]:
        pass
    
    @abstractmethod
    def refresh_plan(self, *, force: bool = False) -> PlanType:
        pass
    
    def mark_plan_dirty(self) -> None:
        self._plan_dirty = True
    
    def is_plan_dirty(self) -> bool:
        return self._plan_dirty
    
    @property
    def latest_plan(self) -> Optional[PlanType]:
        return self._latest_plan


@runtime_checkable
class PartitionPlanProtocol(Protocol):
    fidelity_subsets: Dict[float, Any]
    metadata: Dict[str, Any]


class NoOpPartitioner(Partitioner[Dict[str, Any]]):
    def get_fidelity_subsets(self) -> Dict[float, Any]:
        return {}
    
    def refresh_plan(self, *, force: bool = False) -> Dict[str, Any]:
        return {}


__all__ = ['Partitioner', 'PartitionPlanProtocol', 'NoOpPartitioner']