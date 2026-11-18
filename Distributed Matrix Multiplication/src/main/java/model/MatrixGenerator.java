package model;

import java.util.Random;

public final class MatrixGenerator {
    private MatrixGenerator() {
    }

    public static int[][] generate(int size, long seed) {
        if (size <= 0) {
            throw new IllegalArgumentException("Matrix size must be positive.");
        }

        Random random = new Random(seed);
        int[][] matrix = new int[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = random.nextInt(100);
            }
        }
        return matrix;
    }
}
