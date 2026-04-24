from services.inference_service import SegmentationInferenceService


def run_medical_inference(patient_dict, weights_path, output_dir, device_str="cpu"):
    image_paths = patient_dict.get("image", [])
    if not image_paths:
        raise ValueError("patient_dict must contain key 'image' with modality file paths.")

    service = SegmentationInferenceService(weights_path=weights_path, default_device=device_str)
    return service.run(image_paths=image_paths, output_dir=output_dir, device=device_str)