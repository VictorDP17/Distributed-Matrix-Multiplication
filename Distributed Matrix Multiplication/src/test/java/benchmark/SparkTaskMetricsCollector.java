package benchmark;

import org.apache.spark.executor.TaskMetrics;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerTaskEnd;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public final class SparkTaskMetricsCollector extends SparkListener {
    private volatile CountDownLatch taskLatch = new CountDownLatch(0);
    private final AtomicInteger completedTasks = new AtomicInteger();
    private final AtomicLong executorRunTimeMs = new AtomicLong();
    private final AtomicLong executorCpuTimeNs = new AtomicLong();
    private final AtomicLong executorDeserializeTimeMs = new AtomicLong();
    private final AtomicLong resultSerializationTimeMs = new AtomicLong();
    private final AtomicLong resultSizeBytes = new AtomicLong();

    public synchronized void beginAction(int expectedTasks) {
        if (expectedTasks <= 0) {
            throw new IllegalArgumentException("Expected task count must be positive.");
        }
        completedTasks.set(0);
        executorRunTimeMs.set(0L);
        executorCpuTimeNs.set(0L);
        executorDeserializeTimeMs.set(0L);
        resultSerializationTimeMs.set(0L);
        resultSizeBytes.set(0L);
        taskLatch = new CountDownLatch(expectedTasks);
    }

    @Override
    public void onTaskEnd(SparkListenerTaskEnd taskEnd) {
        TaskMetrics metrics = taskEnd.taskMetrics();
        if (metrics != null) {
            executorRunTimeMs.addAndGet(metrics.executorRunTime());
            executorCpuTimeNs.addAndGet(metrics.executorCpuTime());
            executorDeserializeTimeMs.addAndGet(metrics.executorDeserializeTime());
            resultSerializationTimeMs.addAndGet(metrics.resultSerializationTime());
            resultSizeBytes.addAndGet(metrics.resultSize());
        }
        completedTasks.incrementAndGet();
        taskLatch.countDown();
    }

    public Snapshot awaitSnapshot(long timeout, TimeUnit unit) throws InterruptedException {
        boolean complete = taskLatch.await(timeout, unit);
        return new Snapshot(
                complete,
                completedTasks.get(),
                executorRunTimeMs.get(),
                executorCpuTimeNs.get(),
                executorDeserializeTimeMs.get(),
                resultSerializationTimeMs.get(),
                resultSizeBytes.get());
    }

    public record Snapshot(
            boolean complete,
            int completedTasks,
            long executorRunTimeMs,
            long executorCpuTimeNs,
            long executorDeserializeTimeMs,
            long resultSerializationTimeMs,
            long resultSizeBytes) {
        public static Snapshot unavailable() {
            return new Snapshot(false, 0, -1L, -1L, -1L, -1L, -1L);
        }

        public double executorCpuTimeMs() {
            return executorCpuTimeNs < 0 ? -1.0 : executorCpuTimeNs / 1_000_000.0;
        }
    }
}
