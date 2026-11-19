# Distributed Matrix Multiplication

**Course:** Big Data · Academic year 2026–2027  
**Degree:** Data Science and Engineering · University of Las Palmas de Gran Canaria  
**Research title:** Distributed Matrix Multiplication  
**Student:** Víctor Díaz Puga

## Abstract

This study evaluates a Java matrix-multiplication implementation using Apache Spark and compares it with sequential and shared-memory ForkJoin implementations. The experiment keeps each matrix size fixed while varying the configured local parallelism from 1 to 8 slots. It measures end-to-end multiplication time, estimates JVM heap usage, and records additional Spark execution metrics where available.

The measurements show that ForkJoin benefits substantially from additional threads for the larger tested matrices. Spark's elapsed time generally decreases when more local slots are configured, but Spark remains slower than both CPU implementations for every tested size and configuration. The tested matrices are relatively small, and the experiment uses Spark local mode; therefore, the results describe local framework overhead and local parallel execution, not network or multi-machine scalability.

## 1. Objective

The objective is to investigate how a distributed-computing framework behaves when the available execution parallelism is increased, and to understand the balance between computation and framework overhead.

The main research questions are:

1. How does Apache Spark's end-to-end matrix-multiplication time change when the number of local execution slots increases from 1 to 8?
2. How does the Spark implementation compare with the validated sequential and ForkJoin implementations for the same input matrices?
3. How much of the observed cost is associated with row preparation, broadcast creation, RDD construction, task execution, and result collection?

The initial hypothesis is that Spark's scheduling, task, serialization, and collection overhead will dominate for small matrices. ForkJoin is expected to benefit more directly from additional CPU threads because it operates on shared-memory arrays without the distributed-framework layer.

## 2. Reuse from Assignment 3

The sequential and shared-memory parallel implementations reuse the matrix-multiplication code and input-generation approach developed in Assignment 3, although they have been improved:

[Parallel Matrix Multiplication — Assignment 3](https://github.com/VictorDP17/Parallel-Matrix-Multiplication)

The common arithmetic is the loop-reordered `i-k-j` algorithm. The sequential reference is `MatrixMultiplier.multiplyLoopReordered(A, B)`, while the parallel implementation is `MatrixMultiplier.multiplyParallelLoopReordered(A, B, pool)`, which uses a `ForkJoinPool` and recursively assigns disjoint ranges of output rows to tasks.

The input matrices are generated with Java's `Random`, using seeds `777` and `778` for matrices A and B, respectively. Each element is an integer in the range `[0, 99]`. The matrices use Java `int[][]` storage and integer arithmetic. The same generated A and B matrices are used by all three methods for a given matrix size.

## 3. Implementations

Three methods are measured:

- **Sequential baseline:** the loop-reordered `i-k-j` implementation. It provides the reference result and timing baseline.
- **ForkJoin parallel:** the same row-based computation is divided into recursive tasks executed by a `ForkJoinPool` with the configured parallelism.
- **Spark distributed:** matrix A is represented as indexed rows. The rows are parallelized into Spark partitions, while matrix B is sent as a read-only broadcast variable. Each Spark task computes one or more output rows. The returned rows are collected by the driver, sorted by their original row index, and assembled into the output matrix.

### 3.1 Spark data flow

The Spark implementation follows these steps:

1. Convert the rows of A into serializable `MatrixRow` objects, preserving each row's original index.
2. Create a broadcast variable containing B so that tasks can read the same logical matrix without including it in every row record.
3. Create an RDD from the indexed rows of A using `parallelize(inputRows, partitions)` and define a `map` transformation that computes the corresponding output row.
4. Call `collect()`. This is the action that triggers the lazy RDD computation and returns the result rows to the driver.
5. Sort the collected rows by index and reconstruct the final `int[][]` result.
6. Release the broadcast variable after the multiplication.

Only the rows of A are explicitly partitioned as the input RDD. Matrix B is broadcast. The implementation therefore does not partition both matrices into two-dimensional blocks.

### 3.2 Worker-slot and partition configurations

The tested configurations are:

| Configured local slots (`p`) | Spark master | Spark partitions for the tested sizes |
|---:|---|---:|
| 1 | `local[1]` | 4 |
| 2 | `local[2]` | 8 |
| 4 | `local[4]` | 16 |
| 8 | `local[8]` | 32 |

The benchmark uses `partitions = min(n, max(1, p * 4))`. Since all tested matrix sizes are at least 100, the resulting partition counts are 4, 8, 16, and 32.

In this experiment, `local[p]` means Spark local mode with `p` local execution threads. These values are not Spark worker machines, cluster processes on separate hosts, or physical CPU cores. The worker-slot setting and the partition count both change between configurations, so the experiment compares combined local-parallelism configurations rather than isolating the effect of thread count while holding the partition count constant.

## 4. Requirements and Environment

### Software requirements

- Java JDK 21.
- Apache Maven 3.9.12.
- Apache Spark 4.3.0.
- Spark Core for Scala 2.13 (`spark-core_2.13`).
- Python 3.12.10 for the analysis script.
- Python packages: `pandas` and `matplotlib`.

The Maven project declares Spark Core with `provided` scope. Spark's runtime libraries must therefore be available when the application is launched. Use the IDE's configured Maven runtime classpath or a matching Spark installation with `spark-submit`; a plain `java -jar` invocation does not package the Spark runtime dependencies.

### Hardware and operating system

| Item | Value |
|---|---|
| Operating system | Windows 11 Home 25H2 |
| CPU | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| Physical cores / logical processors | 6 / 12 |
| RAM | 7.65 GB (as recorded in Assignment 3) |
| Java | JDK 21.0.12.1 |
| Maven | 3.9.12 |
| Python | 3.12.10 |
| Apache Spark | 4.3.0 |
| Spark Scala binary version | 2.13 |

The four Spark configurations all ran on this one machine in local mode. No separate physical worker nodes were used.

## 5. Build and Run

### 5.1 Repository layout

The repository is organised as follows:

```text
Distributed-Matrix-Multiplication/
├── Distributed Matrix Multiplication/   # Maven Java project
├── analysis/                            # Python plotter and summarizer
│   └── analyze_results.py
├── results/
│   ├── raw_results.csv                  # Raw benchmark observations
│   ├── summary.csv                      # Generated summary
│   └── plots/                            # Generated figures
└── README.md
```

### 5.2 Compile the Java project

Open a terminal in the Maven project directory:

```bash
cd "Distributed Matrix Multiplication"
mvn clean package
```

The resulting JAR is:

```text
 Distributed Matrix Multiplication/target/Distributed-Matrix-Multiplication-1.0-SNAPSHOT.jar
```

### 5.3 Run the benchmark

The multi-configuration version of `benchmark.BenchmarkMain` accepts the worker-slot list, matrix-size list, repetition count, and output CSV path as arguments.

Run it from the repository root using the IntelliJ IDEA run configuration for `benchmark.BenchmarkMain`, with the Maven dependencies available on the runtime classpath. Set the working directory to the repository root and use these program arguments:

```text
1,2,4,8 100,200,300,400,500 5 results/raw_results.csv
```

If launching outside the IDE, use a Spark 4.3.0 installation so that the `provided` Spark dependencies are available at runtime. From the repository root, the equivalent `spark-submit` invocation is:

```bash
spark-submit --class benchmark.BenchmarkMain \
  "Distributed Matrix Multiplication/target/Distributed-Matrix-Multiplication-1.0-SNAPSHOT.jar" \
  "1,2,4,8" "100,200,300,400,500" "5" "results/raw_results.csv"
```

The application configures the local Spark master for each slot configuration. Do not replace the master setting in the program with a remote cluster address unless the code and experiment are intentionally adapted for a real cluster.

The arguments are:

| Argument | Example | Meaning |
|---|---|---|
| Worker configurations | `1,2,4,8` | Local Spark/ForkJoin parallelism values |
| Matrix sizes | `100,200,300,400,500` | Side lengths of the square matrices |
| Repetitions | `5` | Measured repetitions for each method and configuration |
| Output path | `results/raw_results.csv` | Raw CSV destination |

The program creates a Spark context for each configured local parallelism, runs the experiments, stops that context, and continues with the next configuration. The raw measurements are appended to the same CSV. Remove an old CSV before starting a new experiment if it contains results from a different benchmark version or a different CSV schema.

### 5.4 Generate the summary and plots

From the repository root, run:

```bash
python analysis/analyze_results.py --input results/raw_results.csv --output-dir results
```

The script generates:

- `results/summary.csv`, containing medians, quartiles, IQR, speedup, efficiency, and the median heap estimate.
- `results/plots/`, containing execution-time, speedup, efficiency, and JVM heap-estimate figures for the tested matrix sizes.

The analysis uses the median as the central timing statistic and reports the first and third quartiles. The interquartile range is calculated as `Q3 - Q1`.

## 6. Experimental Methodology

### Inputs and correctness

The tested matrix sizes are `100`, `200`, `300`, `400`, and `500`. For every size, the matrices are generated before the timed multiplication calls with the fixed seeds described in Section 2. Input generation is therefore excluded from the measured multiplication time.

Before recording each result, the benchmark compares the output of the sequential, ForkJoin, and Spark implementations with the sequential reference using an exact integer-matrix equality check. No floating-point tolerance is required because the experiment uses integer arithmetic and the same computation order within each output row.

### Repetitions and warm-up

Each method is warmed up before its measured repetitions. The Spark context is created before the per-multiplication timing and its startup time is reported separately. Five measured repetitions are collected for each method, matrix size, and worker-slot configuration. Warm-up executions are excluded from the raw measured observations.

### Timing boundary

A monotonic `System.nanoTime()` timer is used. The `elapsed_ms` field measures one multiplication call and includes output allocation and computation.

- **Sequential and ForkJoin:** matrix generation and `ForkJoinPool` creation are outside the timed region. The output matrix allocation and multiplication are inside it.
- **Spark:** Spark context startup and input generation are outside the timed region. The measured call includes row preparation, broadcast creation, RDD/transform construction, execution triggered by `collect()`, result collection and assembly, and broadcast cleanup performed by the method.

The Spark warm-up is not included in the measured samples. The timing boundaries are not identical internally: Spark's end-to-end time includes framework and result-collection overhead that is not present in the sequential method. This difference is part of the comparison and must be considered when interpreting the results.

### Strong-scaling design

For each matrix size, the total problem size remains fixed while `p` increases from 1 to 8. This is a local strong-scaling experiment. It does not test weak scaling.

The partition count also increases with `p` (four partitions per configured local slot for the tested sizes). Consequently, this experiment measures the combined effect of more local execution threads and more partitions. A follow-up experiment could hold the partition count fixed while varying `p` to separate these effects.

## 7. Measurements and CSV Columns

The raw CSV contains one row per measured method, repetition, matrix size, and worker-slot configuration.

| Column | Meaning |
|---|---|
| `spark_master` | Spark master string, such as `local[4]` |
| `execution_environment` | `local` for the tested execution mode |
| `configured_worker_slots` | Configured local parallelism `p`; not a physical worker-node count |
| `spark_partitions` | Number of input RDD partitions |
| `matrix_size` | Side length `n` of the square matrices |
| `method` | `sequential_baseline`, `parallel_fork_join`, or `spark_distributed` |
| `repetition` | Measured repetition index |
| `elapsed_ms` | End-to-end multiplication-call time in milliseconds |
| `heap_pool_peak_sum_mb_estimate` | JVM heap-pool peak estimate in MiB |
| `row_preparation_ms` | Time to wrap A's rows as indexed `MatrixRow` objects |
| `broadcast_creation_ms` | Time spent creating the Spark broadcast variable |
| `rdd_construction_ms` | Time to create the parallelized RDD and define the row transformation |
| `collect_and_assembly_ms` | Time covering the `collect()` action, returned rows, sorting, and output-matrix assembly |
| `spark_executor_run_time_sum_ms` | Sum of Spark task executor run times |
| `spark_executor_cpu_time_sum_ms` | Sum of Spark task executor CPU times |
| `spark_task_result_bytes_sum` | Sum of reported task-result sizes in bytes |
| `estimated_broadcast_payload_bytes` | Estimated raw size of B (`n × n × 4` bytes for 32-bit integers) |

The phase timings help describe where time is spent, but they should not all be interpreted as independent wall-clock components. In particular, executor run-time and CPU-time values are sums across tasks; concurrent tasks can overlap, so these sums are not elapsed durations and must not be added to `collect_and_assembly_ms`. The RDD transformation is lazy: the actual task computation is triggered by the `collect()` action.

The broadcast payload is an estimate of the raw integer matrix size, not a measurement of bytes transferred over the network. Task-result sizes are also not a measurement of all network traffic. The experiment does not isolate complete network-transfer costs.

The heap field is an estimate based on JVM heap-pool measurements. It is not process resident memory (RSS), not a per-worker memory measurement, and not a measure of total physical RAM consumption. In local mode, it must be interpreted as a JVM-level estimate rather than memory used by separate worker machines.

## 8. Results and Interpretation

The following table summarises the supplied measurements for the largest tested matrix size (`500 × 500`). Times are medians in milliseconds. The speedup and efficiency columns reproduce the values generated by the current analysis script.

| Local slots (`p`) | Sequential median (ms) | ForkJoin median (ms) | Spark median (ms) | ForkJoin speedup* | Spark speedup* |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.8518 | 42.3102 | 118.4567 | 1.0837 | 0.3871 |
| 2 | 42.0225 | 29.2239 | 86.5861 | 1.4379 | 0.4853 |
| 4 | 44.0620 | 14.9509 | 76.8946 | 2.9471 | 0.5730 |
| 8 | 38.2704 | 7.9476 | 61.0624 | 4.8153 | 0.6267 |

\* **Speedup calculation note:** the current Python script matches each method to the sequential median with the same `configured_worker_slots` value. Thus, the values above are within-configuration comparisons. They are not the conventional fixed-reference strong-scaling speedups based on one common `p = 1` sequential time for every `p`.

### Main observations

- **ForkJoin scales well for larger tested matrices.** For `500 × 500`, its median time falls from 42.31 ms with one slot to 7.95 ms with eight slots. The current script reports a speedup of approximately 4.82 for that configuration. The strongest reported ForkJoin speedup in the supplied data is approximately 5.13 for `400 × 400` with eight slots.
- **Small workloads do not always benefit from additional parallelism.** At `100 × 100`, ForkJoin times are all close to or below one millisecond, where task-splitting and measurement variability can be comparable to the useful computation. The observed speedup is therefore inconsistent at this size.
- **Spark's time decreases overall as more local slots are configured, but the trend is not strictly monotonic for every matrix size.** For `500 × 500`, the median falls from 118.46 ms at one slot to 61.06 ms at eight slots. Nevertheless, Spark remains slower than the sequential and ForkJoin methods in every supplied configuration.
- **Spark overhead is particularly visible for small matrices.** For `100 × 100`, Spark takes tens of milliseconds while the CPU implementations take less than a few milliseconds. At these sizes, framework setup, task scheduling, serialization, and result collection can outweigh the matrix arithmetic.
- **The heap estimate is generally higher for Spark.** For `500 × 500` and eight slots, the supplied estimate is about 329.18 MiB for Spark, compared with about 228.18 MiB for ForkJoin and 227.18 MiB for the sequential method. This is a JVM heap estimate, not per-worker memory or resident memory.

Overall, the results support the expectation that a shared-memory implementation is more suitable for these relatively small dense matrices on a single machine. Spark shows a reduction in elapsed time as local parallelism increases, but its framework overhead prevents it from outperforming the sequential and ForkJoin baselines in the tested range.

## 9. Limitations

- **Local execution only:** all Spark configurations use `local[p]` on one physical machine. The experiment does not demonstrate inter-machine network performance, distributed storage, fault tolerance across hosts, or multi-machine memory capacity. A local multi-thread run should only be presented as the assignment's permitted local alternative if the instructor allows it.
- **Slots are not nodes:** `configured_worker_slots` denotes local Spark execution parallelism, not a count of cluster worker machines or separate executor processes.
- **Partitions also vary:** the number of partitions increases with `p`, so thread parallelism and partitioning are not independently controlled.
- **No complete network measurement:** broadcast payload size is estimated and task-result bytes are collected, but total bytes transferred and network time are not isolated.
- **Memory scope:** the heap-pool estimate is not per-worker memory, RSS, or total system memory.
- **Task metric overlap:** executor time and CPU-time sums aggregate task measurements and can overlap when tasks run concurrently. They are not additional wall-clock phases.
- **Problem sizes:** the tested sizes range only from 100 to 500. They are useful for observing framework overhead but are not large enough to establish how Spark behaves for substantially larger, memory-intensive workloads.
- **Speedup reference:** as described in Section 8, the current analysis joins each method with the sequential measurement from the same slot configuration. A fixed `p = 1` reference should be used for conventional strong-scaling speedup and efficiency.

## 10. Conclusions

This experiment implements dense matrix multiplication with Apache Spark and compares it with sequential and ForkJoin implementations using identical seeded integer matrices. With fixed matrix sizes and local parallelism increased from 1 to 8, ForkJoin achieves a clear reduction in execution time for the larger tested inputs. Spark also becomes faster overall as additional local slots are configured, but it remains slower than both CPU implementations because the tested workloads are small relative to the framework overhead.

The study demonstrates local Spark execution and the costs associated with preparing, scheduling, executing, and collecting a partitioned computation. It does not establish multi-machine distributed scalability. A future experiment should use a real multi-node Spark cluster, larger matrices, per-executor resource measurements, and direct network measurements. It should also hold the partition count fixed in a separate run and use a single `p = 1` sequential reference when calculating conventional strong-scaling speedup and efficiency.

## 11. References and Reproducibility Materials

1. Apache Spark, *RDD Programming Guide*, Spark 4.3.0 documentation: https://downloads.apache.org/spark/docs/4.3.0/rdd-programming-guide.html
2. Apache Spark, *SparkContext Java API*, Spark 4.3.0 documentation: https://downloads.apache.org/spark/docs/4.3.0/api/java/org/apache/spark/SparkContext.html
3. Víctor Díaz Puga, *Parallel Matrix Multiplication (Assignment 3)*, source repository: https://github.com/VictorDP17/Parallel-Matrix-Multiplication

The repository contains the Java source code, Maven configuration, raw benchmark data, and Python analysis script. Keep `raw_results.csv` with the submission so that the summary and figures can be regenerated.

## 12. AI Assistance Disclosure

- **AI tool:** ChatGPT.
- **Purpose:** assistance with README drafting, explanation of the benchmark design, and interpretation of the supplied measurements.
- **Parts affected:** README text and the discussion of the experimental results and limitations.
- **Validation:** the numerical results quoted here were taken from the benchmark summary supplied for this README. The author is responsible for checking the final README against the committed source code, reproducing the commands, and verifying the reported environment before submission.
