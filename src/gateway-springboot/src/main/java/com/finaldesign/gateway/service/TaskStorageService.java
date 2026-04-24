package com.finaldesign.gateway.service;

import com.finaldesign.gateway.config.StorageProperties;
import com.finaldesign.gateway.model.TaskContext;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

@Service
public class TaskStorageService {

    private final Path uploadRoot;
    private final Path outputRoot;
    private final ModalityResolver modalityResolver;

    public TaskStorageService(StorageProperties storageProperties, ModalityResolver modalityResolver) throws IOException {
        this.uploadRoot = Path.of(storageProperties.getUploadRoot()).toAbsolutePath().normalize();
        this.outputRoot = Path.of(storageProperties.getOutputRoot()).toAbsolutePath().normalize();
        this.modalityResolver = modalityResolver;

        Files.createDirectories(this.uploadRoot);
        Files.createDirectories(this.outputRoot);
    }

    public TaskContext createTask(List<MultipartFile> files) throws IOException {
        if (files == null || files.isEmpty()) {
            throw new IllegalArgumentException("No files part in the request.");
        }
        if (files.size() != 4) {
            throw new IllegalArgumentException("Expected 4 NIfTI modalities, but got " + files.size() + ".");
        }

        String taskId = UUID.randomUUID().toString().replace("-", "").substring(0, 8);
        Path uploadDir = uploadRoot.resolve(taskId);
        Path outputDir = outputRoot.resolve(taskId);
        Files.createDirectories(uploadDir);
        Files.createDirectories(outputDir);

        List<Path> savedPaths = new ArrayList<>();
        try {
            for (MultipartFile file : files) {
                String filename = StringUtils.cleanPath(file.getOriginalFilename() == null ? "" : file.getOriginalFilename());
                if (filename.isBlank()) {
                    continue;
                }
                validateNiftiExtension(filename);

                Path target = uploadDir.resolve(filename).normalize();
                if (!target.startsWith(uploadDir)) {
                    throw new IllegalArgumentException("Invalid file name: " + filename);
                }

                try (InputStream inputStream = file.getInputStream()) {
                    Files.copy(inputStream, target, StandardCopyOption.REPLACE_EXISTING);
                }
                savedPaths.add(target);
            }

            if (savedPaths.size() != 4) {
                throw new IllegalArgumentException("Exactly 4 valid files must be provided.");
            }

            List<String> orderedPaths = modalityResolver.orderPaths(savedPaths);
            return new TaskContext(taskId, uploadDir, outputDir, orderedPaths);
        } catch (Exception exception) {
            cleanupDir(uploadDir);
            cleanupDir(outputDir);
            throw exception;
        }
    }

    public Path resolveOutputFile(String taskId, String filename) {
        String cleaned = StringUtils.cleanPath(filename == null ? "" : filename);
        if (cleaned.isBlank()) {
            return null;
        }

        Path filePath = outputRoot.resolve(taskId).resolve(cleaned).normalize();
        Path taskRoot = outputRoot.resolve(taskId).normalize();
        if (!filePath.startsWith(taskRoot) || !Files.exists(filePath)) {
            return null;
        }
        return filePath;
    }

    private static void validateNiftiExtension(String filename) {
        String lower = filename.toLowerCase(Locale.ROOT);
        if (!lower.endsWith(".nii") && !lower.endsWith(".nii.gz")) {
            throw new IllegalArgumentException("Invalid file type: " + filename + ". Only .nii or .nii.gz files are supported.");
        }
    }

    private static void cleanupDir(Path dir) {
        if (dir == null || !Files.exists(dir)) {
            return;
        }
        try {
            Files.walk(dir)
                    .sorted((left, right) -> right.compareTo(left))
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException ignored) {
                        }
                    });
        } catch (IOException ignored) {
        }
    }
}
