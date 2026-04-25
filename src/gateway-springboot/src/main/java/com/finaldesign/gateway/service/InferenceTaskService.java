package com.finaldesign.gateway.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.finaldesign.gateway.model.InferResponse;
import com.finaldesign.gateway.model.TaskContext;
import com.finaldesign.gateway.model.TaskStatusResponse;
import com.finaldesign.gateway.model.TaskSubmitResponse;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Service
public class InferenceTaskService {

    private static final int DEFAULT_POLL_INTERVAL_MS = 1000;

    private final InferenceClient inferenceClient;
    private final ObjectMapper objectMapper;
    private final ConcurrentHashMap<String, TaskRuntime> taskStore = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newFixedThreadPool(Math.max(2, Runtime.getRuntime().availableProcessors() / 2));

    public InferenceTaskService(InferenceClient inferenceClient, ObjectMapper objectMapper) {
        this.inferenceClient = inferenceClient;
        this.objectMapper = objectMapper;
    }

    public TaskSubmitResponse submit(TaskContext taskContext, String model, String requestId) {
        String taskId = taskContext.taskId();
        TaskRuntime runtime = TaskRuntime.queued(taskId, model, requestId);
        taskStore.put(taskId, runtime);

        executor.submit(() -> executeTask(runtime, taskContext));

        return new TaskSubmitResponse(
                "Task accepted",
                taskId,
                runtime.status,
                "/api/tasks/" + taskId,
                DEFAULT_POLL_INTERVAL_MS
        );
    }

    public TaskStatusResponse getStatus(String taskId) {
        TaskRuntime runtime = taskStore.get(taskId);
        if (runtime == null) {
            throw new NoSuchElementException("Task not found: " + taskId);
        }
        return runtime.toResponse();
    }

    private void executeTask(TaskRuntime runtime, TaskContext taskContext) {
        runtime.update("RUNNING", 15, "Validating task and preparing inference context", null, null, null);

        try {
            runtime.update("RUNNING", 45, "Inference in progress", null, null, null);

            InferResponse inferResponse = inferenceClient.infer(
                    taskContext.imagePaths(),
                    taskContext.outputDir().toString(),
                    runtime.model,
                    runtime.requestId
            );

            String maskFilename = inferResponse.mask_filename() == null || inferResponse.mask_filename().isBlank()
                    ? "pred_mask.nii.gz"
                    : inferResponse.mask_filename();

            String maskUrl = "/api/download/" + taskContext.taskId() + "/" + maskFilename;
            String reportFilename = "report.json";
            String reportUrl = "/api/download/" + taskContext.taskId() + "/" + reportFilename;

            runtime.update("RUNNING", 85, "Packaging clinical report", null, maskUrl, reportUrl);
            writeMetricReport(taskContext.outputDir(), taskContext.taskId(), runtime.model, runtime.requestId, inferResponse.metrics());

            runtime.update("COMPLETED", 100, "Inference completed", inferResponse.metrics(), maskUrl, reportUrl);
        } catch (Exception exception) {
            runtime.update("FAILED", 100, "Inference failed", null, null, null);
            runtime.error = exception.getMessage();
            runtime.updatedAt = Instant.now().toString();
        }
    }

    private void writeMetricReport(
            Path outputDir,
            String taskId,
            String model,
            String requestId,
            Map<String, Double> metrics
    ) throws IOException {
        Files.createDirectories(outputDir);

        Instant now = Instant.now();

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schema_version", "1.1");

        Map<String, Object> taskMeta = new LinkedHashMap<>();
        taskMeta.put("task_id", taskId);
        taskMeta.put("model", model);
        taskMeta.put("request_id", requestId);
        taskMeta.put("generated_at", now.toString());
        report.put("task", taskMeta);

        Map<String, Object> clinical = new LinkedHashMap<>();
        clinical.put("unit", "cm3");
        clinical.put("regions", Map.of(
                "necrotic", valueOrNull(metrics, "necrotic_cm3"),
                "edema", valueOrNull(metrics, "edema_cm3"),
                "enhancing", valueOrNull(metrics, "enhancing_cm3"),
                "whole_tumor", valueOrNull(metrics, "total_cm3")
        ));
        clinical.put("raw", metrics);
        report.put("clinical_metrics", clinical);

        Map<String, Object> evaluation = new LinkedHashMap<>();
        evaluation.put("status", "not_available_in_online_inference");
        evaluation.put("note", "Dice/HD95 require ground-truth labels and are exported in offline evaluation reports.");
        evaluation.put("dice", null);
        evaluation.put("hd95_mm", null);
        report.put("evaluation_metrics", evaluation);

        Map<String, Object> artifacts = new LinkedHashMap<>();
        artifacts.put("mask_filename", "pred_mask.nii.gz");
        artifacts.put("report_filename", "report.json");
        report.put("artifacts", artifacts);

        Path reportPath = outputDir.resolve("report.json");
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(reportPath.toFile(), report);
    }

    private static Double valueOrNull(Map<String, Double> metrics, String key) {
        if (metrics == null) {
            return null;
        }
        return metrics.get(key);
    }

    private static final class TaskRuntime {
        private final String taskId;
        private final String model;
        private final String requestId;

        private volatile String status;
        private volatile int progress;
        private volatile String message;
        private volatile Map<String, Double> metrics;
        private volatile String error;
        private volatile String maskUrl;
        private volatile String reportUrl;
        private volatile String updatedAt;

        private TaskRuntime(String taskId, String model, String requestId) {
            this.taskId = taskId;
            this.model = model;
            this.requestId = requestId;
            this.status = "QUEUED";
            this.progress = 0;
            this.message = "Task queued";
            this.metrics = null;
            this.error = null;
            this.maskUrl = null;
            this.reportUrl = null;
            this.updatedAt = Instant.now().toString();
        }

        static TaskRuntime queued(String taskId, String model, String requestId) {
            return new TaskRuntime(taskId, model, requestId);
        }

        void update(
                String status,
                int progress,
                String message,
                Map<String, Double> metrics,
                String maskUrl,
                String reportUrl
        ) {
            this.status = status;
            this.progress = progress;
            this.message = message;
            if (metrics != null) {
                this.metrics = metrics;
            }
            if (maskUrl != null) {
                this.maskUrl = maskUrl;
            }
            if (reportUrl != null) {
                this.reportUrl = reportUrl;
            }
            this.updatedAt = Instant.now().toString();
        }

        TaskStatusResponse toResponse() {
            return new TaskStatusResponse(
                    taskId,
                    status,
                    progress,
                    message,
                    model,
                    maskUrl,
                    reportUrl,
                    metrics,
                    error,
                    updatedAt
            );
        }
    }
}
