from pathlib import Path
import io

import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path(
    "../models/efficientnet_b0_groupaware_finetuned_best.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

class_to_idx = checkpoint["class_to_idx"]

idx_to_class = {
    idx: class_name
    for class_name, idx in class_to_idx.items()
}

IMAGE_SIZE = checkpoint.get(
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


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="HemaVision API",
    description=(
        "Explainable blood-cell classification "
        "using EfficientNet-B0"
    ),
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "HemaVision API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(device),
        "model": "EfficientNet-B0",
        "classes": list(class_to_idx.keys())
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Invalid file"
        )

    if not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file"
        )

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read image"
        )

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

    predicted_class = idx_to_class[
        predicted_idx
    ]

    confidence = probabilities[
        predicted_idx
    ].item()

    class_probabilities = {
        idx_to_class[i]:
            round(
                probabilities[i].item(),
                6
            )

        for i in range(
            len(probabilities)
        )
    }

    return {
        "predicted_class":
            predicted_class,

        "confidence":
            round(confidence, 6),

        "probabilities":
            class_probabilities,

        "disclaimer":
            (
                "Educational and research "
                "use only. Not intended "
                "for clinical diagnosis."
            )
    }