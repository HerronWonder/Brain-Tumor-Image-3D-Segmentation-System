package com.finaldesign.gateway.service;

import org.springframework.stereotype.Component;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

@Component
public class ModalityResolver {

    private static final List<String> MODALITY_ORDER = List.of("t1", "t1ce", "t2", "flair");
    private static final Pattern T1_PATTERN = Pattern.compile("(^|[_\\-.])t1([_\\-.]|$)");
    private static final Pattern T2_PATTERN = Pattern.compile("(^|[_\\-.])t2([_\\-.]|$)");

    public List<String> orderPaths(List<Path> paths) {
        if (paths.size() != 4) {
            throw new IllegalArgumentException("Expected 4 modality files, got " + paths.size() + ".");
        }

        Map<String, Path> resolved = new HashMap<>();
        List<String> unresolved = new ArrayList<>();

        for (Path path : paths) {
            String modality = detectModality(path.getFileName().toString());
            if (modality == null) {
                unresolved.add(path.getFileName().toString());
                continue;
            }
            if (resolved.containsKey(modality)) {
                throw new IllegalArgumentException("Duplicate modality detected for '" + modality + "'.");
            }
            resolved.put(modality, path);
        }

        if (!unresolved.isEmpty()) {
            throw new IllegalArgumentException("Unable to infer modality from file names: " + String.join(", ", unresolved));
        }

        List<String> missing = MODALITY_ORDER.stream().filter(modality -> !resolved.containsKey(modality)).toList();
        if (!missing.isEmpty()) {
            throw new IllegalArgumentException("Missing modality files: " + String.join(", ", missing));
        }

        return MODALITY_ORDER.stream().map(modality -> resolved.get(modality).toString()).toList();
    }

    private String detectModality(String filename) {
        String name = filename.toLowerCase(Locale.ROOT);
        if (name.contains("t1ce") || name.contains("t1c")) {
            return "t1ce";
        }
        if (name.contains("flair")) {
            return "flair";
        }
        if (T2_PATTERN.matcher(name).find()) {
            return "t2";
        }
        if (T1_PATTERN.matcher(name).find()) {
            return "t1";
        }
        return null;
    }
}
