package com.finaldesign.gateway.model;

import java.nio.file.Path;
import java.util.List;

public record TaskContext(
        String taskId,
        Path uploadDir,
        Path outputDir,
        List<String> imagePaths
) {
}
