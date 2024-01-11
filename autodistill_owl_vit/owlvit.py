import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OWLViT(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(
        self,
        ontology: CaptionOntology,
        model: Optional[Union[str, os.PathLike]] = "google/owlvit-base-patch32",
    ):
        self.ontology = ontology
        self.processor = OwlViTProcessor.from_pretrained(model)
        self.model = OwlViTForObjectDetection.from_pretrained(model).to(DEVICE)

    def predict(self, input: str, confidence=0.1) -> sv.Detections:
        labels = self.ontology.prompts()

        image = load_image(input, return_format="PIL")

        with torch.no_grad():
            inputs = self.processor(text=labels, images=image, return_tensors="pt").to(
                DEVICE
            )
            outputs = self.model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])

            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes
            )

            i = 0

            boxes = results[i]["boxes"].tolist()
            scores = results[i]["scores"].tolist()
            labels = results[i]["labels"].tolist()

            # filter with score < confidence
            boxes = [box for box, score in zip(boxes, scores) if score > confidence]
            labels = [
                label for label, score in zip(labels, scores) if score > confidence
            ]
            scores = [score for score in scores if score > confidence]

            if len(boxes) == 0:
                return sv.Detections.empty()

            detections = sv.Detections(
                xyxy=np.array(boxes),
                class_id=np.array(labels),
                confidence=np.array(scores),
            )

            return detections
