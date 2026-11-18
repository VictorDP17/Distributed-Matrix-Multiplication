package benchmark;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;

public final class HeapMemory {
    private HeapMemory() {
    }

    public static void resetHeapPoolPeaks() {
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP) {
                try {
                    pool.resetPeakUsage();
                } catch (UnsupportedOperationException | IllegalArgumentException ignored) {
                }
            }
        }
    }

    public static double heapPoolPeakSumMb() {
        long bytes = 0L;
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP && pool.getPeakUsage() != null) {
                bytes += Math.max(0L, pool.getPeakUsage().getUsed());
            }
        }
        return bytes / (1024.0 * 1024.0);
    }
}
