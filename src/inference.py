from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model(model_path, device):

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }

    image_size = checkpoint.get(
        "image_size",
        224
    )

    model = efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_to_idx)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return (
        model,
        idx_to_class,
        image_size
    )


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def create_transform(image_size):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(
    model,
    image_path,
    transform,
    idx_to_class,
    device
):

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_idx = torch.argmax(
        probabilities
    ).item()

    predicted_class = (
        idx_to_class[
            predicted_idx
        ]
    )

    confidence = (
        probabilities[
            predicted_idx
        ].item()
    )

    class_probabilities = {
        idx_to_class[i]:
            probabilities[i].item()

        for i in range(
            len(probabilities)
        )
    }

    return (
        predicted_class,
        confidence,
        class_probabilities
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "HemaVision Blood Cell "
            "Classification Inference"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained .pth model"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to microscope image"
    )

    args = parser.parse_args()

    model_path = Path(
        args.model
    )

    image_path = Path(
        args.image
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{model_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    (
        model,
        idx_to_class,
        image_size
    ) = load_model(
        model_path,
        device
    )

    transform = create_transform(
        image_size
    )

    (
        predicted_class,
        confidence,
        class_probabilities
    ) = predict_image(
        model,
        image_path,
        transform,
        idx_to_class,
        device
    )

    print("\n" + "=" * 50)

    print(
        "HEMAVISION PREDICTION"
    )

    print("=" * 50)

    print(
        f"\nPrediction: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        "\nClass Probabilities:"
    )

    for class_name, probability in sorted(
        class_probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{class_name:<15} "
            f"{probability * 100:>7.2f}%"
        )

    print("\n" + "=" * 50)

    print(
        "Educational/research use only."
    )

    print(
        "Not intended for clinical diagnosis."
    )


if __name__ == "__main__":
    main()