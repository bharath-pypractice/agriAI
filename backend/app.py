import os
import traceback

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps
from tensorflow.keras.layers import DepthwiseConv2D  # type: ignore[import]
from tensorflow.keras.models import load_model  # type: ignore[import]


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

np.set_printoptions(suppress=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


print("Loading TensorFlow model...")
model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
)
print("Model loaded successfully.")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]


fertilizer_recommendations = {
    "Tomato_Healthy": {
        "fertilizer": "NPK 10-10-10",
        "treatment": "Maintain proper watering and sunlight",
    },
    "Tomato_Early_Blight": {
        "fertilizer": "Copper Fungicide",
        "treatment": "Spray fungicide every 7 days",
    },
    "Tomato_Late_Blight": {
        "fertilizer": "Mancozeb",
        "treatment": "Apply fungicide immediately",
    },
    "Tomato_Leaf_Spot": {
        "fertilizer": "Chlorothalonil",
        "treatment": "Spray weekly",
    },
    "Rice_Bacterial_Blight": {
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Apply Streptomycin spray",
    },
    "Rice_Brown_Spot": {
        "fertilizer": "Mancozeb",
        "treatment": "Spray fungicide weekly",
    },
    "Rice_Leaf_Blast": {
        "fertilizer": "Tricyclazole",
        "treatment": "Spray fungicide every 7 days",
    },
    "Rice_Sheath_Blight": {
        "fertilizer": "Validamycin",
        "treatment": "Apply fungicide during early infection",
    },
}


def _extract_disease_name(class_label: str) -> str:
    if " " in class_label:
        class_label = class_label.split(" ", 1)[1]
    return class_label


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty file name"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = int(np.argmax(prediction))
        class_name = class_names[index]

        confidence_score = float(prediction[0][index])
        disease = _extract_disease_name(class_name)
        confidence_percent = round(confidence_score * 100)
        disease_display = disease.replace("_", " ")

        if confidence_percent < 65:
            return jsonify(
                {
                    "prediction": disease_display,
                    "confidence": confidence_percent,
                    "disease": disease_display,
                    "suggestions": [],
                    "low_confidence": True,
                    "message": (
                        "The AI is not confident about this image (below 65%). "
                        "Please upload a clearer close-up of the affected leaves, "
                        "or consult a nearby agriculture officer/engineer."
                    ),
                }
            )

        rec = fertilizer_recommendations.get(
            disease,
            {
                "fertilizer": "General NPK fertilizer",
                "treatment": "Consult local agricultural expert",
            },
        )

        return jsonify(
            {
                "prediction": disease_display,
                "confidence": confidence_percent,
                "disease": disease_display,
                "suggestions": [rec["fertilizer"], rec["treatment"]],
                "low_confidence": False,
            }
        )

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print("Prediction Error:", e)
        print(tb)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)