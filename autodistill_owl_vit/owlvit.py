import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.detection.caption_ontology import CaptionOntology
from autodistill.detection.detection_base_model import DetectionBaseModel

# , DetectionBaseModel, DetectionOntology
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @dataclass
class OWLViT(DetectionBaseModel):
    # ontology: CaptionOntology
    # owlvit_model: model

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-large-patch14"
        ).to(DEVICE)
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")

    def predict(self, input: str) -> sv.Detections:
        labels = self.ontology.prompts()

        image = Image.open(input).convert("RGB")

        with torch.no_grad():
            if isinstance(self.ontology, CaptionOntology):
                inputs = self.processor(
                    text=labels, images=image, return_tensors="pt"
                ).to(DEVICE)
                outputs = self.model(**inputs)
            elif isinstance(self.ontology, DetectionOntology):
                inputs = self.processor(
                    query_images=labels, images=image, return_tensors="pt"
                ).to(DEVICE)
                outputs = self.model.image_guided_detection(**inputs)
                outputs["pred_boxes"] = outputs["target_pred_boxes"]

            target_sizes = torch.Tensor([image.size[::-1]]).to(DEVICE)

            results = self.processor.post_process(
                outputs=outputs, target_sizes=target_sizes
            )

            results = [
                {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in results
            ]

            i = 0

            detections = sv.Detections(
                xyxy=np.array(results[i]["boxes"]),
                class_id=np.array(results[i]["labels"]),
                confidence=np.array(results[i]["scores"]),
            )

            return detections
