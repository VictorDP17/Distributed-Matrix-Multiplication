package benchmark;

import distributed.SparkMatrixMultiplier;
import model.MatrixGenerator;
import model.MatrixMultiplier;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;

public final class CorrectnessTest {
    private CorrectnessTest() {
    }

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf()
                .setAppName("Assignment-Correctness")
                .setMaster("local[2]")
                .set("spark.ui.enabled", "false");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        sparkContext.sc().setLogLevel("ERROR");

        try {
            checkKnownProduct(sparkContext);
            checkIdentityAndZero(sparkContext);
            checkGeneratedMatrices(sparkContext);
            System.out.println("All Assignment correctness checks passed.");
        } finally {
            sparkContext.stop();
        }
    }

    private static void checkKnownProduct(JavaSparkContext sparkContext) {
        int[][] a = {{1, 2}, {3, 4}};
        int[][] b = {{5, 6}, {7, 8}};
        int[][] expected = {{19, 22}, {43, 50}};
        assertEqual(expected, MatrixMultiplier.multiplyLoopReordered(a, b), "known sequential product");
        assertEqual(expected, SparkMatrixMultiplier.multiply(sparkContext, a, b, 2), "known Spark product");
    }

    private static void checkIdentityAndZero(JavaSparkContext sparkContext) {
        int[][] a = {{2, 3, 4}, {5, 6, 7}, {8, 9, 1}};
        int[][] identity = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        int[][] zero = new int[3][3];
        assertEqual(a, SparkMatrixMultiplier.multiply(sparkContext, a, identity, 2), "identity property");
        assertEqual(zero, SparkMatrixMultiplier.multiply(sparkContext, a, zero, 2), "zero property");
    }

    private static void checkGeneratedMatrices(JavaSparkContext sparkContext) {
        int[][] a = MatrixGenerator.generate(40, 777L);
        int[][] b = MatrixGenerator.generate(40, 778L);
        int[][] expected = MatrixMultiplier.multiplyLoopReordered(a, b);
        try (ForkJoinPool pool = new ForkJoinPool(2)) {
            assertEqual(expected, MatrixMultiplier.multiplyParallelLoopReordered(a, b, pool),
                    "ForkJoin comparison");
        }
        for (int partitions : new int[]{1, 2, 4, 8}) {
            assertEqual(expected, SparkMatrixMultiplier.multiply(sparkContext, a, b, partitions),
                    "Spark comparison with " + partitions + " partitions");
        }
    }

    private static void assertEqual(int[][] expected, int[][] actual, String context) {
        if (!Arrays.deepEquals(expected, actual)) {
            throw new IllegalStateException("Correctness check failed: " + context);
        }
    }
}
