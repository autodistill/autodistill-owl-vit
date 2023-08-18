<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill OWL-ViT Module

This repository contains the code supporting the OWL-ViT base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) is a transformer-based object detection model developed by Google Research.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [OWL-ViT Autodistill documentation](https://autodistill.github.io/autodistill/base_models/owlvit/).

## Installation

To use OWL-ViT with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-owl-vit
```

## Quickstart

```python
from autodistill_owl_vit import OWLViT

# define an ontology to map class names to our OWLViT prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = OWLViT(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpg")
```

To use a query images instead of query text, use a DetectionOntology instead of a CaptionOntology:

```python
from autodistill.detection import DetectionOntology

from autodistill_owl_vit import OWLViT

example_cat = PIL.Image.open("cat.jpg")

base_model = OWLViT(ontology=DetectionOntology([(example_cat, "cat")]))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpg")
```

## License

The code in this repository is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!