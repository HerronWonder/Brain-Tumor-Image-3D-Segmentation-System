from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    image_paths: list[str] = Field(min_length=4, max_length=4)
    output_dir: str
    model: str = "unet"


class InferResponse(BaseModel):
    mask_filename: str
    model: str
    metrics: dict[str, float]
