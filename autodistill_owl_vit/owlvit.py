import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from PIL import Image
from autodistill.detection import CaptionOntology, DetectionBaseModel
from transformers import OwlViTProcessor, OwlViTForObjectDetection

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(DEVICE)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

@dataclass
class OWLViT(DetectionBaseModel):
    ontology: CaptionOntology
    owlvit_model: model

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.owlvit_model = model

    def predict(self, input: str) -> sv.Detections:
        labels = self.ontology.prompts()

        image = Image.open(input)

        with torch.no_grad():
            inputs = processor(text=labels, images=image, return_tensors="pt")
            outputs = model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])

            results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

            i = 0

            detections = sv.Detections(
                xyxy=np.array(results[i]["boxes"]),
                class_id=np.array(results[i]["labels"]),
                confidence=np.array(results[i]["scores"]),
            )

            return detections