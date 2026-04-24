package com.finaldesign.gateway.model;

import java.util.Map;

public record InferResponse(
        String mask_filename,
        String model,
        Map<String, Double> metrics
) {
}
