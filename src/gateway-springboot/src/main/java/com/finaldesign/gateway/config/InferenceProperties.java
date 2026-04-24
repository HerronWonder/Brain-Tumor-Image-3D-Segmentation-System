package com.finaldesign.gateway.config;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties(prefix = "inference")
public class InferenceProperties {

    @NotBlank
    private String baseUrl = "http://127.0.0.1:8000";

    @NotBlank
    private String inferPath = "/internal/infer";

    @Min(1)
    private int timeoutSeconds = 600;

    private String internalToken = "";

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    public String getInferPath() {
        return inferPath;
    }

    public void setInferPath(String inferPath) {
        this.inferPath = inferPath;
    }

    public int getTimeoutSeconds() {
        return timeoutSeconds;
    }

    public void setTimeoutSeconds(int timeoutSeconds) {
        this.timeoutSeconds = timeoutSeconds;
    }

    public String getInternalToken() {
        return internalToken;
    }

    public void setInternalToken(String internalToken) {
        this.internalToken = internalToken;
    }
}
