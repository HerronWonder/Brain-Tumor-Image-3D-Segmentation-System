package com.finaldesign.gateway.model;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;

import java.util.List;

public record InferRequest(
        @NotEmpty List<String> image_paths,
        @NotBlank String output_dir,
        @NotBlank String model
) {
}
