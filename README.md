# MFTune

MFTune is a multi-fidelity optimization framework for **Spark SQL configuration tuning**. Its goal is to find high-performance configurations under limited tuning budgets.

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


### Search Space

In MFTune, **search space** means the set of Spark configuration knobs that can be tuned, and the valid value range of each knob.

Full-space dimension table:

| Dimension | Type | Range / Choices | Default |
| --- | --- | --- | --- |
| `spark.task.cpus` | `integer` | `[1, 8]` | `6` |
| `spark.locality.wait` | `integer` | `[0, 10]` | `5` |
| `spark.executor.memory` | `integer` | `[1, 180]` | `93` |
| `spark.executor.cores` | `integer` | `[1, 32]` | `14` |
| `spark.executor.instances` | `integer` | `[1, 24]` | `9` |
| `spark.executor.memoryOverhead` | `integer` | `[384, 20480]` | `16133` |
| `spark.driver.cores` | `integer` | `[1, 16]` | `1` |
| `spark.driver.memory` | `integer` | `[10, 120]` | `75` |
| `spark.default.parallelism` | `integer` | `[100, 3000]` | `495` |
| `spark.sql.shuffle.partitions` | `integer` | `[100, 3000]` | `357` |
| `spark.sql.autoBroadcastJoinThreshold` | `integer` | `[10, 3000]` | `515` |
| `spark.network.timeout` | `integer` | `[120, 30000]` | `14839` |
| `spark.sql.broadcastTimeout` | `integer` | `[300, 30000]` | `8574` |
| `spark.sql.sources.parallelPartitionDiscovery.parallelism` | `integer` | `[10, 500]` | `69` |
| `spark.driver.maxResultSize` | `integer` | `[2048, 6144]` | `3484` |
| `spark.driver.memoryOverhead` | `integer` | `[384, 20480]` | `8266` |
| `spark.reducer.maxSizeInFlight` | `integer` | `[1, 300]` | `191` |
| `spark.shuffle.file.buffer` | `integer` | `[1, 300]` | `42` |
| `spark.shuffle.unsafe.file.output.buffer` | `integer` | `[1, 300]` | `201` |
| `spark.shuffle.spill.diskWriteBufferSize` | `integer` | `[1048576, 104857600]` | `50250071` |
| `spark.shuffle.service.index.cache.size` | `integer` | `[1, 300]` | `270` |
| `spark.shuffle.accurateBlockThreshold` | `integer` | `[1048576, 314572800]` | `48673750` |
| `spark.shuffle.registration.timeout` | `integer` | `[1000, 10000]` | `4111` |
| `spark.shuffle.registration.maxAttempts` | `integer` | `[1, 5]` | `1` |
| `spark.shuffle.mapOutput.minSizeForBroadcast` | `integer` | `[100, 3000]` | `1959` |
| `spark.io.compression.snappy.blockSize` | `integer` | `[1, 96]` | `92` |
| `spark.kryoserializer.buffer.max` | `integer` | `[1, 1024]` | `949` |
| `spark.kryoserializer.buffer` | `integer` | `[1, 300]` | `45` |
| `spark.memory.offHeap.size` | `integer` | `[0, 10]` | `5` |
| `spark.storage.unrollMemoryThreshold` | `integer` | `[1048576, 8388608]` | `4743675` |
| `spark.storage.localDiskByExecutors.cacheSize` | `integer` | `[100, 3000]` | `698` |
| `spark.broadcast.blockSize` | `integer` | `[1, 32]` | `7` |
| `spark.executor.heartbeatInterval` | `integer` | `[5, 100]` | `91` |
| `spark.files.fetchTimeout` | `integer` | `[1, 300]` | `119` |
| `spark.files.maxPartitionBytes` | `integer` | `[10485760, 524288000]` | `330115300` |
| `spark.files.openCostInBytes` | `integer` | `[1048576, 10485760]` | `9186861` |
| `spark.storage.memoryMapThreshold` | `integer` | `[1, 10]` | `4` |
| `spark.network.timeoutInterval` | `integer` | `[30, 600]` | `217` |
| `spark.scheduler.maxRegisteredResourcesWaitingTime` | `integer` | `[10, 120]` | `18` |
| `spark.scheduler.revive.interval` | `integer` | `[1, 10]` | `2` |
| `spark.scheduler.excludeOnFailure.unschedulableTaskSetTimeout` | `integer` | `[100, 600]` | `220` |
| `spark.speculation.interval` | `integer` | `[100, 1000]` | `255` |
| `spark.task.maxFailures` | `integer` | `[1, 10]` | `1` |
| `spark.task.reaper.pollingInterval` | `integer` | `[5, 60]` | `49` |
| `spark.stage.maxConsecutiveAttempts` | `integer` | `[1, 10]` | `5` |
| `spark.sql.files.maxPartitionBytes` | `integer` | `[10485760, 524288000]` | `101434143` |
| `spark.speculation.quantile` | `float` | `[0.1, 1]` | `0.25` |
| `spark.memory.fraction` | `float` | `[0.1, 0.9]` | `0.30000000000000004` |
| `spark.memory.storageFraction` | `float` | `[0.1, 0.9]` | `0.9` |
| `spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor` | `float` | `[0.1, 1.0]` | `0.1` |
| `spark.io.compression.codec` | `categorical` | `lz4`, `snappy`, `zstd` | `lz4` |
| `spark.serializer` | `categorical` | `org.apache.spark.serializer.JavaSerializer`, `org.apache.spark.serializer.KryoSerializer` | `org.apache.spark.serializer.JavaSerializer` |
| `spark.shuffle.compress` | `categorical` | `true`, `false` | `true` |
| `spark.shuffle.spill.compress` | `categorical` | `true`, `false` | `true` |
| `spark.rdd.compress` | `categorical` | `true`, `false` | `false` |
| `spark.speculation` | `categorical` | `true`, `false` | `false` |
| `spark.sql.adaptive.enabled` | `categorical` | `true`, `false` | `true` |
| `spark.sql.adaptive.coalescePartitions.enabled` | `categorical` | `true`, `false` | `true` |
| `spark.sql.adaptive.skewJoin.enabled` | `categorical` | `true`, `false` | `true` |
| `spark.scheduler.minRegisteredResourcesRatio` | `float` | `[0.5, 1.0]` | `0.55` |


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
