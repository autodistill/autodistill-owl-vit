import os

# from dataclasses import dataclass
import numpy as np
import supervision as sv
import torch
from autodistill.detection import DetectionBaseModel
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

# from autodistill_owl_vit.owl_vit_object_detection.src.models import OwlViT
from .models import OwlViT

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(
#     DEVICE
# )
# checkpoint = torch.load("/home/jovyan/speir-datavol-115/models/owl_models/test.pt")
# model = OwlViT().load_state_dict(checkpoint["model"]).to(DEVICE)
# model = OwlViT().load_state_dict(torch.load('state_dict_path')).to(DEVICE)
#     torch.load("/home/jovyan/speir-datavol-115/models/owl_models/test.pt")
# ).to(DEVICE)


# @dataclass
class OWLViT_Finetuned(DetectionBaseModel):
    # ontology: CaptionOntology
    # owlvit_model: model

    def __init__(self, model_path, labelmap):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
        pretrained_model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-large-patch14"
        )

        to_encode = []
        for label in labelmap.values():
            to_encode.append(label)
            to_encode.append("a photo of " + label)
            to_encode.append("a " + label + " in an environment")

        inputs = self.processor(
            text=[to_encode],
            images=Image.new("RGB", (224, 224)),
            return_tensors="pt",
        )
        with torch.no_grad():
            queries = pretrained_model(**inputs).text_embeds

        self.ft_model = (
            OwlViT(pretrained_model=pretrained_model, query_bank=queries)
            .load_state_dict(torch.load(model_path))
            .to(DEVICE)
        )

    def predict(self, input: str) -> sv.Detections:
        # labels = self.ontology.prompts()

        image = Image.open(input).convert("RGB")

        with torch.no_grad():
            # if isinstance(self.ontology, CaptionOntology):
            # inputs = processor(text=labels, images=image, return_tensors="pt").to(
            #     DEVICE
            # )
            image = self.processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
            # pred_boxes in 'corners' format ?
            pred_boxes, pred_class_logits, pred_class_sims, _ = self.ft_model(image)
            # # else:
            #     print("Not implemented")
            #     exit()

            # target_sizes = torch.Tensor([image.size[::-1]]).to(DEVICE)

            # results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

            # results = [
            #     {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in results
            # ]

            # i = 0

            confidences = softmax(pred_class_logits)

            detections = sv.Detections(
                xyxy=np.array(pred_boxes),
                class_id=np.array(pred_class_sims),
                confidence=np.array(confidences),
            )

            return detections


def softmax(logits):
    e = np.exp(logits - np.max(logits))  # to prevent overflow
    return e / e.sum(axis=-1, keepdims=True)
