import io
import base64
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
from flask import Flask, request, jsonify, render_template

# --------- Model definition (same as training) ----------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # logits

# --------- Load model ----------
device = torch.device("cpu")
model = MNISTCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def preprocess_pil(pil_img: Image.Image) -> torch.Tensor:
    """
    Converts a canvas PNG (usually black digit on white background)
    into MNIST-like tensor: (1, 1, 28, 28) normalized.
    Includes centering by cropping ink bbox and padding.
    """
    # 1) Grayscale
    img = pil_img.convert("L")

    # 2) Invert: canvas black-on-white -> MNIST-like white-on-black
    img = ImageOps.invert(img)

    # 3) Convert to numpy [0,255] for bbox detection
    arr255 = np.array(img)

    # 4) Find "ink" pixels (anything above a small threshold)
    # Threshold avoids tiny noise being treated as ink
    ys, xs = np.where(arr255 > 30)

    # If nothing drawn, fall back to plain resize
    if len(xs) == 0 or len(ys) == 0:
        img28 = img.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(img28).astype(np.float32) / 255.0
        arr = (arr - MNIST_MEAN) / MNIST_STD
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # 5) Crop to bounding box of ink
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1)) #use + 1 to include y_max or x_max pixel itself

    # 6) Resize cropped digit to fit into a 20x20 box (MNIST-ish convention)
    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(20 * h / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(20 * w / h)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 7) Paste into centered 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)  # black background
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(resized, (left, top))

    # 8) To tensor + normalize
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image"}), 400

        b64 = data["image"].split(",")[-1]
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")

        x = preprocess_pil(pil_img).to(device)

        with torch.no_grad():
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            conf = float(probs[pred])

        return jsonify({"prediction": pred, "confidence": conf})

    except Exception as e:
        print("PREDICT ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)