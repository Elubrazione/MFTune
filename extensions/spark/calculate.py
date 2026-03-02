from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import numpy as np
import math
from scipy.stats import spearmanr, pearsonr, kendalltau
from openbox.utils.history import History, Observation
from openbox import logger


@dataclass
class ProcessedHistory:
    task_id: str
    weight: float
    normalized_weight: float
    objectives: np.ndarray  # shape: (n_configs,)
    sql_times: Dict[str, np.ndarray] = field(default_factory=dict)  # sql_name -> times, shape: (n_configs,)
    
    @property
    def n_configs(self) -> int:
        return len(self.objectives)
    
    def get_subset_times(self, sql_names: List[str]) -> np.ndarray:
        if not sql_names:
            return np.zeros(self.n_configs)
        
        subset_times = np.zeros(self.n_configs)
        for sql_name in sql_names:
            if sql_name in self.sql_times:
                subset_times += self.sql_times[sql_name]
            else:
                raise ValueError(f"SQL {sql_name} not found in ProcessedHistory")
        return subset_times


def preprocess_histories(
    histories_with_weights: Sequence[Tuple[History, float]],
    sql_columns: Sequence[str],
    sql_type: str = "qt",
) -> List[ProcessedHistory]:
    valid_histories = [(h, w) for h, w in histories_with_weights if w > 0 and len(h) > 0]
    if not valid_histories:
        return []
    
    weights = np.array([w for _, w in valid_histories])
    total_weight = weights.sum()
    normalized_weights = weights / total_weight if total_weight > 0 else weights
    
    processed_list: List[ProcessedHistory] = []
    
    for (history, weight), norm_weight in zip(valid_histories, normalized_weights):
        objectives_list: List[float] = []
        sql_times_dict: Dict[str, List[float]] = {sql: [] for sql in sql_columns}
        
        for obs in history.observations:
            obs_objectives = getattr(obs, "objectives", None)
            if not obs_objectives or not np.isfinite(obs_objectives[0]):
                continue
            
            objective = float(obs_objectives[0])
            
            extra_info = getattr(obs, "extra_info", None) or {}
            obs_sql_times: Dict[str, float] = extra_info.get(f"{sql_type}_time", {})
            
            sql_times_for_config: Dict[str, float] = {}
            all_valid = True
            for sql_name in sql_columns:
                sql_time = obs_sql_times.get(sql_name, float("inf"))
                if np.isfinite(sql_time):
                    sql_times_for_config[sql_name] = sql_time
                else:
                    all_valid = False
                    break
            
            if not all_valid:
                continue
            
            objectives_list.append(objective)
            for sql_name in sql_columns:
                sql_times_dict[sql_name].append(sql_times_for_config[sql_name])
        
        if len(objectives_list) < 3:
            continue
        
        processed = ProcessedHistory(
            task_id=getattr(history, "task_id", "unknown"),
            weight=weight,
            normalized_weight=norm_weight,
            objectives=np.array(objectives_list, dtype=float),
            sql_times={sql: np.array(times, dtype=float) for sql, times in sql_times_dict.items()},
        )
        processed_list.append(processed)
    
    if processed_list:
        total_weight = sum(p.weight for p in processed_list)
        logger.info(f"Processed {len(processed_list)} histories, total_weight={total_weight:.4f}")
        if total_weight > 0:
            for p in processed_list:
                p.normalized_weight = p.weight / total_weight
            logger.info(f"All processed histories:")
            for i, p in enumerate(processed_list):
                logger.info(f"  [{i}] task_id={p.task_id}, weight={p.weight:.4f}, normalized_weight={p.normalized_weight:.4f}, "
                      f"#objectives={len(p.objectives)}, #sqls={len(p.sql_times)}")
    
    return processed_list


def _compute_correlation(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    try:
        if method == "spearman":
            corr, _ = spearmanr(x, y)
        elif method == "pearson":
            corr, _ = pearsonr(x, y)
        elif method == "kendall":
            corr, _ = kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def compute_correlation_within_history(
    processed: ProcessedHistory,
    sql_names: List[str],
    correlation_method: str = "spearman",
) -> float:
    """
    Calculate the correlation between the subset times and the objectives within a single history.
    
    Correlation = sum(correlation_within_history_i * history_i's normalized weight)
    """
    
    subset_times = processed.get_subset_times(sql_names)
    corr = _compute_correlation(subset_times, processed.objectives, correlation_method)
    return abs(corr) 


def compute_weighted_correlation(
    processed_histories: List[ProcessedHistory],
    sql_names: List[str],
    correlation_method: str = "spearman",
) -> float:
    weighted_corr = 0.0
    for processed in processed_histories:
        corr = compute_correlation_within_history(processed, sql_names, correlation_method)
        weighted_corr += corr * processed.normalized_weight
    return weighted_corr



def compute_sql_estimated_times(
    processed_histories: List[ProcessedHistory],
    sql_columns: List[str],
) -> Dict[str, float]:
    if not processed_histories or not sql_columns:
        return {}
    
    total_weighted_objective = sum(
        np.mean(p.objectives) * p.normalized_weight 
        for p in processed_histories
    )
    
    sql_estimated_times: Dict[str, float] = {}
    
    for sql_name in sql_columns:
        weighted_avg_time = sum(
            np.mean(p.sql_times.get(sql_name, np.zeros(p.n_configs))) * p.normalized_weight
            for p in processed_histories
        )
        estimated_time = weighted_avg_time / total_weighted_objective if total_weighted_objective > 0 else 0.0
        sql_estimated_times[sql_name] = estimated_time
        
        logger.info(f"SQL {sql_name}: estimated_time={estimated_time:.4f}")
    
    total_estimated_time = sum(sql_estimated_times.values())
    logger.info(f"Sum of all estimated_times: {total_estimated_time:.6f} (should be close to 1.0)")
    
    return sql_estimated_times


def greedy_select_subset(
    processed_histories: List[ProcessedHistory],
    sql_estimated_times: Dict[str, float],
    budget_ratio: float,
    used_queries: List[str],
    correlation_method: str = "spearman",
    tolerance: float = 0.1,
) -> Tuple[List[str], float, float]:
    if not processed_histories or not sql_estimated_times:
        return [], 0.0
    
    total_estimated_time = sum(sql_estimated_times.values())
    budget = budget_ratio * total_estimated_time
    max_budget = budget * (1 + tolerance)
    logger.info(f"Greedy selection: budget_ratio={budget_ratio}, budget={budget:.4f}, max_budget={max_budget:.4f}")
    
    candidates = [sql for sql in sql_estimated_times.keys() if sql not in used_queries]
    selected: List[str] = []
    current_time = 0.0
    final_corr = 0.0 
    
    while current_time < max_budget and candidates:
        best_sql = None
        best_corr = -float("inf")
        
        for sql_name in candidates:
            estimated_time = sql_estimated_times[sql_name]
            if current_time + estimated_time > max_budget:
                continue
            
            test_subset = selected + [sql_name]
            subset_corr = compute_weighted_correlation(processed_histories, test_subset, correlation_method)
            if subset_corr > best_corr:
                best_sql = sql_name
                best_corr = subset_corr
    
        if best_sql is None:
            break
        
        selected.append(best_sql)
        current_time += sql_estimated_times[best_sql]
        candidates.remove(best_sql)
        final_corr = best_corr
        logger.info(f"  Selected {best_sql}: subset_corr={best_corr:.4f}, current_time={current_time:.4f}")
    
    actual_ratio = current_time / total_estimated_time if total_estimated_time > 0 else 0.0
    logger.info(f"Greedy selection done: {len(selected)} SQLs, actual_ratio={actual_ratio:.4f}, final_corr={final_corr:.4f}")
    
    return selected, actual_ratio, final_corr


def multi_fidelity_sql_selection(
    histories_with_weights: Sequence[Tuple[History, float]],
    fidelity_levels: Sequence[float],
    sql_columns: Sequence[str],
    *,
    correlation_method: str = "spearman",
    sql_type: str = "qt",
    tolerance: float = 0.1,
) -> Tuple[Dict[float, List[str]], Dict[str, float], Dict[float, float]]:
    sql_columns = list(sql_columns)
    
    if not histories_with_weights or not sql_columns:
        return {}, {}, {}
    
    processed_histories = preprocess_histories(histories_with_weights, sql_columns, sql_type)
    if not processed_histories:
        logger.warning("No valid histories after preprocessing")
        return {}, {}, {}
    
    
    sql_estimated_times = compute_sql_estimated_times(processed_histories, sql_columns)
    if not sql_estimated_times:
        logger.warning("No SQL estimated times computed")
        return {}, {}, {}
    
    fidelity_subsets: Dict[float, List[str]] = {}
    fidelity_correlations: Dict[float, float] = {}
    used_queries: List[str] = []
    
    for fidelity in sorted(fidelity_levels):
        logger.info(f"Fidelity level: {fidelity}")
        
        if math.isclose(fidelity, 1.0):
            fidelity_subsets[fidelity] = sql_columns.copy()
            fidelity_correlations[fidelity] = 1.0
            logger.info(f"  Full set: {len(sql_columns)} SQLs, subset_corr={1.0:.4f}")
            continue
        
        selected, actual_ratio, subset_corr = greedy_select_subset(
            processed_histories,
            sql_estimated_times,
            fidelity,
            used_queries,
            correlation_method,
            tolerance,
        )
        
        fidelity_subsets[fidelity] = selected
        used_queries.extend(selected)
        
        fidelity_correlations[fidelity] = subset_corr
        logger.info(f"  Selected {len(selected)} SQLs, subset_corr={subset_corr:.4f}")
    
    return fidelity_subsets, sql_estimated_times, fidelity_correlations
