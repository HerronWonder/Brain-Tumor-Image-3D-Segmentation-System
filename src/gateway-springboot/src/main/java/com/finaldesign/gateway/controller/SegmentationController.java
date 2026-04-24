package com.finaldesign.gateway.controller;

import com.finaldesign.gateway.model.InferResponse;
import com.finaldesign.gateway.model.PredictResponse;
import com.finaldesign.gateway.model.TaskContext;
import com.finaldesign.gateway.service.InferenceClient;
import com.finaldesign.gateway.service.TaskStorageService;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api")
public class SegmentationController {

    private final TaskStorageService taskStorageService;
    private final InferenceClient inferenceClient;

    public SegmentationController(TaskStorageService taskStorageService, InferenceClient inferenceClient) {
        this.taskStorageService = taskStorageService;
        this.inferenceClient = inferenceClient;
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "running", "service", "Brain Tumor Segmentation Gateway"));
    }

    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<PredictResponse> predict(
            @RequestPart("files") List<MultipartFile> files,
            @RequestParam(name = "model", defaultValue = "unet") String model,
            @RequestHeader(name = "X-Request-ID", required = false) String incomingRequestId
    ) throws IOException {
        String normalizedModel = model.trim().toLowerCase(Locale.ROOT);
        String requestId = StringUtils.hasText(incomingRequestId)
            ? incomingRequestId.trim()
            : UUID.randomUUID().toString().replace("-", "");
        TaskContext task = taskStorageService.createTask(files);

        InferResponse inferResponse = inferenceClient.infer(
                task.imagePaths(),
                task.outputDir().toString(),
            normalizedModel,
            requestId
        );

        String maskFilename = inferResponse.mask_filename() == null || inferResponse.mask_filename().isBlank()
                ? "pred_mask.nii.gz"
                : inferResponse.mask_filename();

        PredictResponse response = new PredictResponse(
                "Inference successful",
                task.taskId(),
                "/api/download/" + task.taskId() + "/" + maskFilename,
                inferResponse.model(),
                inferResponse.metrics()
        );

        return ResponseEntity.ok()
            .header("X-Request-ID", requestId)
            .body(response);
    }

    @GetMapping("/download/{taskId}/{filename}")
    public ResponseEntity<Resource> download(
            @PathVariable String taskId,
            @PathVariable String filename
    ) {
        Path filePath = taskStorageService.resolveOutputFile(taskId, filename);
        if (filePath == null) {
            return ResponseEntity.notFound().build();
        }

        Resource resource = new FileSystemResource(filePath);
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=" + resource.getFilename())
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(resource);
    }
}
