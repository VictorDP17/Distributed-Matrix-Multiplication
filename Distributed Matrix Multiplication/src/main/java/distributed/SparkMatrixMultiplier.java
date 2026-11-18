package distributed;

import model.MatrixMultiplier;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public final class SparkMatrixMultiplier {
    private SparkMatrixMultiplier() {
    }

    public static int[][] multiply(JavaSparkContext sparkContext, int[][] a, int[][] b, int requestedPartitions) {
        return multiplyDetailed(sparkContext, a, b, requestedPartitions).matrix();
    }

    public static DetailedResult multiplyDetailed(JavaSparkContext sparkContext, int[][] a, int[][] b, int requestedPartitions) {
        int n = validateInput(sparkContext, a, b, requestedPartitions);

        TimedResult<List<MatrixRow>> preparedRows = prepareRows(a, n);

        int partitions = Math.min(n, requestedPartitions);

        TimedResult<Broadcast<int[][]>> broadcastResult =
                createBroadcast(sparkContext, b);

        Broadcast<int[][]> broadcastB = broadcastResult.result();

        try {
            TimedResult<JavaRDD<MatrixRow>> distributedResult =
                    distributeAndMultiply(
                            sparkContext,
                            preparedRows.result(),
                            broadcastB,
                            partitions
                    );

            TimedResult<int[][]> collectedResult =
                    collectAndAssemble(distributedResult.result(), n);

            long estimatedBroadcastPayloadBytes =
                    (long) n * n * Integer.BYTES;

            return new DetailedResult(
                    collectedResult.result(),
                    preparedRows.elapsedMs(),
                    broadcastResult.elapsedMs(),
                    distributedResult.elapsedMs(),
                    collectedResult.elapsedMs(),
                    estimatedBroadcastPayloadBytes
            );

        } finally {
            broadcastB.unpersist(true);
        }
    }

    private static int validateInput(JavaSparkContext sparkContext, int[][] a, int[][] b, int requestedPartitions) {
        int n = MatrixMultiplier.validateAndGetSize(a, b);

        if (sparkContext == null) {
            throw new IllegalArgumentException(
                    "A JavaSparkContext is required."
            );
        }

        if (requestedPartitions <= 0) {
            throw new IllegalArgumentException(
                    "The partition count must be positive."
            );
        }

        return n;
    }

    private static TimedResult<List<MatrixRow>> prepareRows(int[][] a, int n) {
        long start = System.nanoTime();

        List<MatrixRow> inputRows = new ArrayList<>(n);

        for (int i = 0; i < n; i++) {
            inputRows.add(new MatrixRow(i, a[i]));
        }

        return new TimedResult<>(
                inputRows,
                nanosToMillis(System.nanoTime() - start)
        );
    }

    private static TimedResult<Broadcast<int[][]>> createBroadcast(JavaSparkContext sparkContext, int[][] b) {
        long start = System.nanoTime();

        Broadcast<int[][]> broadcastB = sparkContext.broadcast(b);

        return new TimedResult<>(
                broadcastB,
                nanosToMillis(System.nanoTime() - start)
        );
    }

    private static TimedResult<JavaRDD<MatrixRow>> distributeAndMultiply(
            JavaSparkContext sparkContext,
            List<MatrixRow> inputRows,
            Broadcast<int[][]> broadcastB,
            int partitions) {

        long start = System.nanoTime();

        JavaRDD<MatrixRow> distributedRows =
                sparkContext.parallelize(inputRows, partitions);

        JavaRDD<MatrixRow> resultRows = distributedRows.map(row ->
                new MatrixRow(
                        row.getIndex(),
                        multiplySingleRow(
                                row.getValues(),
                                broadcastB.value()
                        )
                )
        );

        return new TimedResult<>(
                resultRows,
                nanosToMillis(System.nanoTime() - start)
        );
    }

    private static TimedResult<int[][]> collectAndAssemble(JavaRDD<MatrixRow> resultRows, int n) {
        long start = System.nanoTime();

        List<MatrixRow> collectedRows = new ArrayList<>(resultRows.collect());

        if (collectedRows.size() != n) {
            throw new IllegalStateException(
                    "Expected " + n + " output rows, received "
                            + collectedRows.size() + "."
            );
        }

        collectedRows.sort(
                Comparator.comparingInt(MatrixRow::getIndex)
        );

        int[][] result = new int[n][n];

        for (int i = 0; i < n; i++) {
            MatrixRow row = collectedRows.get(i);

            if (row.getIndex() != i || row.getValues().length != n) {
                throw new IllegalStateException(
                        "Spark returned a missing or malformed output row."
                );
            }

            result[i] = row.getValues();
        }

        return new TimedResult<>(
                result,
                nanosToMillis(System.nanoTime() - start)
        );
    }

    private record TimedResult<T>(T result, double elapsedMs) {
    }

    private static int[] multiplySingleRow(int[] aRow, int[][] b) {
        int n = aRow.length;
        int[] resultRow = new int[n];
        for (int k = 0; k < n; k++) {
            int aik = aRow[k];
            for (int j = 0; j < n; j++) {
                resultRow[j] += aik * b[k][j];
            }
        }
        return resultRow;
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    public record DetailedResult(
            int[][] matrix,
            double rowPreparationMs,
            double broadcastCreationMs,
            double rddConstructionMs,
            double collectAndAssemblyMs,
            long estimatedBroadcastPayloadBytes) {
    }
}
