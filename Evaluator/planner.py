from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generic, TypeVar, Callable
from openbox import logger

PlanType = TypeVar('PlanType')


class Planner(ABC, Generic[PlanType]):
    
    def __init__(self):
        self._cached_plan: Optional[PlanType] = None
    
    @abstractmethod
    def plan(
        self, 
        resource_ratio: float, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        pass
    
    def refresh_plan(self, *, force: bool = False) -> Optional[PlanType]:
        return None
    
    def _ensure_plan(
        self, 
        *, 
        force_refresh: bool = False,
        check_dirty: Optional[Callable[[], bool]] = None,
    ) -> Optional[PlanType]:
        if force_refresh:
            logger.debug("Force refreshing planner plan")
            return self.refresh_plan(force=True)
        
        if self._cached_plan is None:
            logger.debug("No cached plan, refreshing")
            return self.refresh_plan(force=True)
        
        if check_dirty is not None and check_dirty():
            logger.debug("Planner plan is dirty, refreshing")
            return self.refresh_plan(force=True)
        
        return self._cached_plan


class NoOpPlanner(Planner[None]):
    def plan(self, resource_ratio: float, **kwargs) -> Optional[Dict[str, Any]]:
        return None


__all__ = ['Planner', 'NoOpPlanner']