package distributed;

import java.io.Serial;
import java.io.Serializable;

public final class MatrixRow implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final int index;
    private final int[] values;

    public MatrixRow(int index, int[] values) {
        this.index = index;
        this.values = values;
    }

    public int getIndex() {
        return index;
    }

    public int[] getValues() {
        return values;
    }
}
