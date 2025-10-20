from transformers import Sam2Processor, Sam2Model, infer_device
import torch
from PIL import Image
import requests


if __name__ == "__main__":
    device = infer_device()

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
    processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

    image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
    input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

    inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

    # The model outputs multiple mask predictions ranked by quality score
    print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")
    raw_image.show()
