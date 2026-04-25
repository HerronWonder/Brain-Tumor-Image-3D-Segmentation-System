package com.finaldesign.gateway.model;

public record TaskSubmitResponse(
        String message,
        String task_id,
        String status,
        String status_url,
        int poll_interval_ms
) {
}
