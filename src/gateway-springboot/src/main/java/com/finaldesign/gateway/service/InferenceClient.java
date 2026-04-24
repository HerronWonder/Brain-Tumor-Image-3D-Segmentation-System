package com.finaldesign.gateway.service;

import com.finaldesign.gateway.config.InferenceProperties;
import com.finaldesign.gateway.model.InferRequest;
import com.finaldesign.gateway.model.InferResponse;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.List;

@Service
public class InferenceClient {

    private final WebClient inferenceWebClient;
    private final InferenceProperties inferenceProperties;

    public InferenceClient(WebClient inferenceWebClient, InferenceProperties inferenceProperties) {
        this.inferenceWebClient = inferenceWebClient;
        this.inferenceProperties = inferenceProperties;
    }

    public InferResponse infer(List<String> imagePaths, String outputDir, String model, String requestId) {
        InferRequest request = new InferRequest(imagePaths, outputDir, model);

        InferResponse response = inferenceWebClient
                .post()
                .uri(inferenceProperties.getInferPath())
                .contentType(MediaType.APPLICATION_JSON)
                .headers(headers -> {
                    headers.set("X-Request-ID", requestId);
                    if (StringUtils.hasText(inferenceProperties.getInternalToken())) {
                        headers.set("X-Internal-Token", inferenceProperties.getInternalToken());
                    }
                })
                .bodyValue(request)
                .retrieve()
                .bodyToMono(InferResponse.class)
                .block(Duration.ofSeconds(inferenceProperties.getTimeoutSeconds()));

        if (response == null) {
            throw new IllegalStateException("Inference service returned an empty response.");
        }
        return response;
    }
}
