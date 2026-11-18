package model;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public final class MatrixMultiplier {
    private MatrixMultiplier() {
    }

    public static int[][] multiplyLoopReordered(int[][] a, int[][] b) {
        int n = validateAndGetSize(a, b);
        int[][] result = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                int aik = a[i][k];
                for (int j = 0; j < n; j++) {
                    result[i][j] += aik * b[k][j];
                }
            }
        }
        return result;
    }

    public static int[][] multiplyParallelLoopReordered(int[][] a, int[][] b, ForkJoinPool pool) {
        int n = validateAndGetSize(a, b);
        if (pool == null || pool.isShutdown()) {
            throw new IllegalArgumentException("A running ForkJoinPool is required.");
        }

        int[][] result = new int[n][n];
        int threshold = Math.max(1, n / Math.max(1, pool.getParallelism() * 4));
        pool.invoke(new RowMultiplyTask(a, b, result, 0, n, threshold));
        return result;
    }

    public static int validateAndGetSize(int[][] a, int[][] b) {
        if (a == null || b == null || a.length == 0 || a.length != b.length) {
            throw new IllegalArgumentException("Matrices must be non-empty and have matching dimensions.");
        }
        int n = a.length;
        for (int i = 0; i < n; i++) {
            if (a[i] == null || b[i] == null || a[i].length != n || b[i].length != n) {
                throw new IllegalArgumentException("Only square n x n matrices are supported.");
            }
        }
        return n;
    }

    private static void multiplyRows(int[][] a, int[][] b, int[][] result, int startRow, int endRow) {
        int n = a.length;
        for (int i = startRow; i < endRow; i++) {
            for (int k = 0; k < n; k++) {
                int aik = a[i][k];
                for (int j = 0; j < n; j++) {
                    result[i][j] += aik * b[k][j];
                }
            }
        }
    }

    private static final class RowMultiplyTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;

        private final int[][] a;
        private final int[][] b;
        private final int[][] result;
        private final int startRow;
        private final int endRow;
        private final int threshold;

        private RowMultiplyTask(int[][] a, int[][] b, int[][] result, int startRow, int endRow, int threshold) {
            this.a = a;
            this.b = b;
            this.result = result;
            this.startRow = startRow;
            this.endRow = endRow;
            this.threshold = threshold;
        }

        @Override
        protected void compute() {
            if (endRow - startRow <= threshold) {
                multiplyRows(a, b, result, startRow, endRow);
                return;
            }

            int middle = startRow + (endRow - startRow) / 2;
            invokeAll(
                    new RowMultiplyTask(a, b, result, startRow, middle, threshold),
                    new RowMultiplyTask(a, b, result, middle, endRow, threshold));
        }
    }
}
