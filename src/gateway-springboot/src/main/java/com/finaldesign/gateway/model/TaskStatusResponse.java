package com.finaldesign.gateway.model;

import java.util.Map;

public record TaskStatusResponse(
        String task_id,
        String status,
        int progress,
        String message,
        String model,
        String mask_url,
        String report_url,
        Map<String, Double> metrics,
        String error,
        String updated_at
) {
}
