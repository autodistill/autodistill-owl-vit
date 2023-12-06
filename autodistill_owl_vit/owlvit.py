import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(
    DEVICE
)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")


@dataclass
class OWLViT(DetectionBaseModel):
    ontology: CaptionOntology
    owlvit_model: model

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.owlvit_model = model

    def predict(self, input: str, confidence = 0.1) -> sv.Detections:
        labels = self.ontology.prompts()

        image = load_image(input, return_format="PIL")

        with torch.no_grad():
            inputs = processor(text=labels, images=image, return_tensors="pt")
            outputs = model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])

            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)

            i = 0

            boxes = results[i]["boxes"].tolist()
            scores = results[i]["scores"].tolist()
            labels = results[i]["labels"].tolist()

            print(scores)

            # filter with score < confidence
            boxes = [box for box, score in zip(boxes, scores) if score > confidence]
            labels = [label for label, score in zip(labels, scores) if score > confidence]
            scores = [score for score in scores if score > confidence]

            if len(boxes) == 0:
                return sv.Detections.empty()

            detections = sv.Detections(
                xyxy=np.array(boxes),
                class_id=np.array(labels),
                confidence=np.array(scores),
            )

            return detections
