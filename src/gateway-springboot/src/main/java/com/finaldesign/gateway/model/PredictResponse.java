package com.finaldesign.gateway.model;

import java.util.Map;

public record PredictResponse(
        String message,
        String task_id,
        String mask_url,
        String model,
        Map<String, Double> metrics
) {
}
