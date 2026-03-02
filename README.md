# MFTune

MFTune is a multi-fidelity optimization framework for **Spark SQL configuration tuning**.  
Its goal is to find high-performance configurations under limited tuning budgets.

Key capabilities:
- Transfer learning from historical workloads
- Search-space compression via importance and density estimation
- Multi-fidelity scheduling for low-cost candidate filtering
- Two-phase warm-start for faster convergence


## Code Mapping

### Overview

MFTune first identifies historical workloads similar to the target task, then compresses the original configuration space.  
Next, MFBO generates promising candidates, and a multi-fidelity scheduler evaluates them progressively from low fidelity to full fidelity, filtering poor configurations early.  
After each iteration, new observations are written back to history so the surrogate and recommendation quality keep improving.

### Paper Module ↔ Code Module

| Method module (paper view) | Role | Code module (repo) | Key entry |
| --- | --- | --- | --- |
| Controller | Orchestrates config loading, plugin setup, task initialization, and optimization loop | `main.py`, `manager/` | `main.py` |
| Knowledge Database | Stores task metadata and historical observations | `manager/history_manager.py`, `manager/task_manager.py` | `TaskManager` |
| Similarity Identification | Computes workload similarity from meta-features and historical behavior | `Advisor/workload_mapping/`, `extensions/spark/adapter.py` | `SparkTargetSystem.get_meta_feature()` |
| Search-Space Optimizer | Selects important knobs and narrows value ranges | `Compressor/dimensio/` (git submodule) | `SHAPDimensionStep`, `KDEBoundaryRangeStep` |
| Configuration Generator | Samples candidates via surrogate + acquisition, with transfer and warm-start | `Advisor/` | `Advisor/BO.py`, `Advisor/MFBO.py` |
| Multi-Fidelity Optimizer | Allocates budgets across fidelity levels and eliminates candidates stage by stage | `Optimizer/`, `Optimizer/scheduler/` | `BaseOptimizer.run_one_iter()` |
| Evaluator | Executes Spark SQL evaluations and returns objective values | `Evaluator/`, `extensions/spark/evaluator.py` | `SparkEvaluatorManager` |


**Note:**
1. The paper-level SQL-subset fidelity partition is implemented collaboratively by scheduling and evaluation components, mainly in `extensions/spark/calculate.py`, and `extensions/spark/partitioner.py`. 
2. `Compressor/` is our self-developed standalone package and is integrated here as a git submodule. Please initialize submodules before running experiments: `git submodule update --init --recursive`.



## Repository Structure

```text
MFTune/
├── main.py                  # Main tuning entry
├── manager/                 # Config/task/history management
├── Optimizer/               # Optimization loop and schedulers
├── Advisor/                 # BO/MFBO, surrogates, acquisitions, transfer, warm-start
├── Evaluator/               # Spark evaluation, partitioning, and execution planning
├── extensions/spark/        # Spark plugin (loaded through plugin mechanism)
├── Compressor/              # Dimension filtering and range compression (our self-developed package, integrated as a git submodule)
├── configs/                 # YAML configs and search-space definitions
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.8+ (used in this project: 3.8.20)
- Spark 3.4.x (used in this project: 3.4.1)
- Accessible Spark SQL cluster for real runs

Install dependencies:

```bash
pip install -r requirements.txt
```

## Bash Examples
### TPC-DS Benchmark

```bash
python main.py \
  --opt MFES_SMAC \
  --target tpcds_600g \
  --task 64u256n3 \
  --database tpcds_600g \
  --data_dir /path/to/tpcds/sqls \
  --history_dir /path/to/history \
  --transfer reacq \
  --warm_start best_all \
  --cp_topk 30 \
  --R 9 \
  --eta 3 \
  --use_flatten_scheduler \
  --use_cached_model
```

### TPC-H Benchmark

```bash
python main.py \
  --opt MFES_SMAC \
  --target tpch_600g \
  --task 64u256n3 \
  --database tpch_600g \
  --data_dir /path/to/tpch/sqls \
  --history_dir /path/to/history \
  --transfer reacq \
  --warm_start best_all \
  --cp_topk 30 \
  --R 9 \
  --eta 3 \
  --use_flatten_scheduler \
  --use_cached_model
```

### Key Arguments

- `--config`: YAML config path (default: `configs/base.yaml`)
- `--opt`: Optimization method, e.g. `MFES_SMAC`, `BOHB_*`, `SMAC`
- `--iter_num`: Number of optimization iterations
- `--task`: Task identifier (recommended to encode hardware, e.g. `64u256n3`)
- `--target`: Workload name (used in logging and output paths)
- `--database`: Database name in spark-sql
- `--data_dir`: SQL directory
- `--history_dir`: Historical observations directory
- `--warm_start`: Warm-start strategy (`none` / `best_all` / `random`)
- `--transfer`: Transfer strategy (`reacq`)
- `--cp_topk`: Number of knobs to keep after compression
- `--R`, `--eta`: Multi-fidelity scheduler hyperparameters
- `--use_flatten_scheduler`: Enable expanded full-fidelity bracket scheduling
- `--use_cached_model`: Reuse existing CatBoost model if available


## Configuration Files

Fields you usually need to customize:
- `paths.data_dir` / `paths.history_dir` / `paths.target`
- `database`
- `method_args.cp_args` / `method_args.ws_args` / `method_args.tl_args`
