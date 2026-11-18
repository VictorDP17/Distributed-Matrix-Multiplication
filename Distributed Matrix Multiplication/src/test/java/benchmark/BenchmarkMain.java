package benchmark;

import distributed.SparkMatrixMultiplier;
import model.MatrixGenerator;
import model.MatrixMultiplier;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

public final class BenchmarkMain {
    private static final long SEED_A = 777L;
    private static final long SEED_B = 778L;
    private static final String CSV_HEADER = String.join(",",
            "spark_master",
            "execution_environment",
            "configured_worker_slots",
            "spark_partitions",
            "matrix_size",
            "method",
            "repetition",
            "elapsed_ms",
            "heap_pool_peak_sum_mb_estimate",
            "row_preparation_ms",
            "broadcast_creation_ms",
            "rdd_construction_ms",
            "collect_and_assembly_ms",
            "spark_executor_run_time_sum_ms",
            "spark_executor_cpu_time_sum_ms",
            "spark_task_result_bytes_sum",
            "estimated_broadcast_payload_bytes"
    );

    private BenchmarkMain() {
    }

    public static void main(String[] args) throws Exception {
        int configuredWorkers = args.length > 0 ? parsePositiveInt(args[0], "workers") : 1;
        int[] sizes = args.length > 1 ? parseSizes(args[1]) : new int[]{100, 200, 300, 400, 500};
        int repetitions = args.length > 2 ? parsePositiveInt(args[2], "repetitions") : 5;
        Path output = args.length > 3 ? Path.of(args[3]) : Path.of("results", "raw_results.csv");

        long contextStart = System.nanoTime();
        SparkConf sparkConf = new SparkConf()
                .setAppName("Distributed-Matrix-Multiplication")
                .setMaster("local[*]");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        double contextStartupMs = nanosToMillis(System.nanoTime() - contextStart);
        sparkContext.sc().setLogLevel("WARN");

        String master = sparkContext.sc().master();
        String environment = master.startsWith("local") ? "local" : "cluster_or_remote";
        SparkTaskMetricsCollector taskMetricsCollector = new SparkTaskMetricsCollector();
        sparkContext.sc().addSparkListener(taskMetricsCollector);

        System.out.println("Distributed matrix multiplication benchmark");
        System.out.println("Spark master: " + master);
        System.out.println("Environment: " + environment);
        System.out.println("Configured worker slots: " + configuredWorkers);
        System.out.println("Matrix sizes: " + Arrays.toString(sizes));
        System.out.println("Measured repetitions: " + repetitions);
        System.out.printf("JavaSparkContext startup: %.3f ms (reported separately)%n", contextStartupMs);

        try (ForkJoinPool pool = new ForkJoinPool(configuredWorkers)) {
            for (int n : sizes) {
                System.out.printf("%nMatrix size: %d x %d%n", n, n);
                int[][] a = MatrixGenerator.generate(n, SEED_A);
                int[][] b = MatrixGenerator.generate(n, SEED_B);

                int partitions = Math.min(n, Math.max(1, configuredWorkers * 4));
                int[][] expected = MatrixMultiplier.multiplyLoopReordered(a, b);

                int[][] sequentialWarmup = MatrixMultiplier.multiplyLoopReordered(a, b);
                assertEqual(expected, sequentialWarmup, "sequential warm-up");
                int[][] parallelWarmup = MatrixMultiplier.multiplyParallelLoopReordered(a, b, pool);
                assertEqual(expected, parallelWarmup, "parallel warm-up");

                taskMetricsCollector.beginAction(partitions);
                HeapMemory.resetHeapPoolPeaks();
                long distributedWarmupStart = System.nanoTime();
                SparkMatrixMultiplier.DetailedResult distributedWarmup =
                        SparkMatrixMultiplier.multiplyDetailed(sparkContext, a, b, partitions);
                double distributedWarmupMs = nanosToMillis(System.nanoTime() - distributedWarmupStart);
                taskMetricsCollector.awaitSnapshot(30, TimeUnit.SECONDS);
                assertEqual(expected, distributedWarmup.matrix(), "Spark warm-up");

                List<String> rows = new ArrayList<>(repetitions * 3);
                for (int repetition = 1; repetition <= repetitions; repetition++) {
                    HeapMemory.resetHeapPoolPeaks();
                    long start = System.nanoTime();
                    int[][] sequentialResult = MatrixMultiplier.multiplyLoopReordered(a, b);
                    double elapsedMs = nanosToMillis(System.nanoTime() - start);
                    double memoryMb = HeapMemory.heapPoolPeakSumMb();
                    assertEqual(expected, sequentialResult, "sequential repetition " + repetition);
                    rows.add(csvRow(
                            master, environment,
                            configuredWorkers, partitions, n,
                            "sequential_baseline",
                            repetition, elapsedMs, memoryMb,
                            null, SparkTaskMetricsCollector.Snapshot.unavailable()
                    ));

                    HeapMemory.resetHeapPoolPeaks();
                    start = System.nanoTime();
                    int[][] parallelResult = MatrixMultiplier.multiplyParallelLoopReordered(a, b, pool);
                    elapsedMs = nanosToMillis(System.nanoTime() - start);
                    memoryMb = HeapMemory.heapPoolPeakSumMb();
                    assertEqual(expected, parallelResult, "parallel repetition " + repetition);
                    rows.add(csvRow(
                            master, environment,
                            configuredWorkers, partitions, n,
                            "parallel_fork_join",
                            repetition, elapsedMs, memoryMb,
                            null, SparkTaskMetricsCollector.Snapshot.unavailable()
                    ));

                    taskMetricsCollector.beginAction(partitions);
                    HeapMemory.resetHeapPoolPeaks();
                    start = System.nanoTime();
                    SparkMatrixMultiplier.DetailedResult distributedResult =
                            SparkMatrixMultiplier.multiplyDetailed(sparkContext, a, b, partitions);
                    elapsedMs = nanosToMillis(System.nanoTime() - start);
                    memoryMb = HeapMemory.heapPoolPeakSumMb();
                    SparkTaskMetricsCollector.Snapshot taskMetrics =
                            taskMetricsCollector.awaitSnapshot(30, TimeUnit.SECONDS);
                    assertEqual(expected, distributedResult.matrix(), "Spark repetition " + repetition);
                    rows.add(csvRow(
                            master, environment,
                            configuredWorkers, partitions, n,
                            "spark_distributed",
                            repetition, elapsedMs, memoryMb,
                            distributedResult, taskMetrics
                    ));
                }
                appendCsv(output, rows);
                System.out.printf("Completed %d repetitions for n=%d. Partitions=%d, Spark warm-up=%.3f ms%n",
                        repetitions, n, partitions, distributedWarmupMs);
            }
        } finally {
            sparkContext.sc().removeSparkListener(taskMetricsCollector);
            sparkContext.stop();
        }

        System.out.println("Raw measurements appended to: " + output.toAbsolutePath());
    }

    private static String csvRow(
            String master,
            String environment,
            int configuredWorkers,
            int partitions,
            int matrixSize,
            String method,
            int repetition,
            double elapsedMs,
            double memoryMb,
            SparkMatrixMultiplier.DetailedResult detail,
            SparkTaskMetricsCollector.Snapshot metrics) {

        boolean isSpark = detail != null;

        return String.join(",",
                escapeCsv(master),
                escapeCsv(environment),
                Integer.toString(configuredWorkers),
                Integer.toString(partitions),
                Integer.toString(matrixSize),
                escapeCsv(method),
                Integer.toString(repetition),
                Double.toString(elapsedMs),
                Double.toString(memoryMb),
                isSpark ? Double.toString(detail.rowPreparationMs()) : "-1",
                isSpark ? Double.toString(detail.broadcastCreationMs()) : "-1",
                isSpark ? Double.toString(detail.rddConstructionMs()) : "-1",
                isSpark ? Double.toString(detail.collectAndAssemblyMs()) : "-1",
                Long.toString(metrics.executorRunTimeMs()),
                Double.toString(metrics.executorCpuTimeMs()),
                Long.toString(metrics.resultSizeBytes()),
                isSpark ? Long.toString(detail.estimatedBroadcastPayloadBytes()) : "-1"
        );
    }

    private static String escapeCsv(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    private static void appendCsv(Path output, List<String> rows) throws IOException {
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        boolean writeHeader = Files.notExists(output) || Files.size(output) == 0;
        try (BufferedWriter writer = Files.newBufferedWriter(
                output,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND,
                StandardOpenOption.WRITE)) {
            if (writeHeader) {
                writer.write(CSV_HEADER);
                writer.newLine();
            }
            for (String row : rows) {
                writer.write(row);
                writer.newLine();
            }
        }
    }

    private static int[] parseSizes(String value) {
        String[] tokens = value.split(",");
        int[] sizes = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            sizes[i] = parsePositiveInt(tokens[i].trim(), "matrix size");
        }
        return sizes;
    }

    private static int parsePositiveInt(String value, String label) {
        try {
            int parsed = Integer.parseInt(value);
            if (parsed <= 0) {
                throw new IllegalArgumentException(label + " must be positive.");
            }
            return parsed;
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException("Invalid " + label + ": " + value, exception);
        }
    }

    private static void assertEqual(int[][] expected, int[][] actual, String context) {
        if (!Arrays.deepEquals(expected, actual)) {
            throw new IllegalStateException("Correctness validation failed: " + context);
        }
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }
}
