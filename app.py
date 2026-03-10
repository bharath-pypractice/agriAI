# Install required libraries
# pip install flask tensorflow pillow numpy

import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from flask import Flask, request, jsonify
from google import genai # type: ignore[import]
from google.genai.types import Content, Part # type: ignore[import]

from tensorflow.keras.models import load_model # type: ignore[import]
from tensorflow.keras.layers import DepthwiseConv2D # type: ignore[import]
from PIL import Image, ImageOps
import numpy as np
import traceback
import time

# Change to project root (one level above this file) so .env is found
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env in the project root, if present
dotenv_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

app = Flask(__name__, static_folder='plant-disease-frontend', static_url_path='/')


def _build_gemini_client() -> genai.Client | None:
    """Create a Gemini client if API key is configured, otherwise return None."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

# Disable scientific notation
np.set_printoptions(suppress=True)

# -------------------------------
# Fix Teachable Machine Model
# -------------------------------

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)

# -------------------------------
# Load AI Model
# -------------------------------

print("Loading model...")

model = load_model(
    "keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
)

print("Model loaded successfully")

# -------------------------------
# Load Labels
# -------------------------------

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -------------------------------
# Fertilizer Database
# -------------------------------

fertilizer_recommendations = {

    "Tomato_Healthy": {
        "fertilizer": "NPK 10-10-10",
        "treatment": "Maintain proper watering and sunlight"
    },

    "Tomato_Early_Blight": {
        "fertilizer": "Copper Fungicide",
        "treatment": "Spray fungicide every 7 days"
    },

    "Tomato_Late_Blight": {
        "fertilizer": "Mancozeb",
        "treatment": "Apply fungicide immediately"
    },

    "Tomato_Leaf_Spot": {
        "fertilizer": "Chlorothalonil",
        "treatment": "Spray weekly"
    },

    "Rice_Bacterial_Blight": {
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Apply Streptomycin spray"
    },

    "Rice_Brown_Spot": {
        "fertilizer": "Mancozeb",
        "treatment": "Spray fungicide weekly"
    },

    "Rice_Leaf_Blast": {
        "fertilizer": "Tricyclazole",
        "treatment": "Spray fungicide every 7 days"
    },

    "Rice_Sheath_Blight": {
        "fertilizer": "Validamycin",
        "treatment": "Apply fungicide during early infection"
    }

}


def _build_disease_context(disease: str, suggestions: list[str]) -> str:
    joined_suggestions = "; ".join(suggestions)
    return (
        f"Disease: {disease}. "
        f"Recommendations from the vision model: {joined_suggestions}. "
        "Explain things clearly for a farmer with simple agricultural terms. "
        "Be practical and concise."
    )


def _looks_like_overloaded_gemini_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("503" in msg and "unavailable" in msg) or ("unavailable" in msg) or ("rate limit" in msg) or ("429" in msg)


def _gemini_generate_with_retries(
    client: genai.Client,
    *,
    model: str,
    contents: list[Content],
    max_attempts: int = 4,
    initial_backoff_s: float = 0.8,
) -> object:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            retryable = _looks_like_overloaded_gemini_error(e)
            if not retryable or attempt == max_attempts:
                raise
            sleep_s = initial_backoff_s * (2 ** (attempt - 1))
            time.sleep(sleep_s)
    raise last_exc or RuntimeError("Gemini request failed.")


# -------------------------------
# Homepage
# -------------------------------

@app.route('/')
def index():
    return app.send_static_file('index.html')

# -------------------------------
# Disease Detection API
# -------------------------------

@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    try:

        # Read image
        image = Image.open(file.stream).convert("RGB")

        # Resize
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Convert to numpy
        image_array = np.asarray(image)

        # Normalize
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Create model input
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Prediction
        prediction = model.predict(data)

        index = np.argmax(prediction)
        class_name = class_names[index]

        confidence_score = float(prediction[0][index])

        # Remove label number
        if " " in class_name:
            disease = class_name.split(" ", 1)[1]
        else:
            disease = class_name

        confidence_percent = round(confidence_score * 100)

        # If confidence is too low, don't give fertilizer suggestions.
        if confidence_percent < 65:
            return jsonify({
                "disease": disease.replace("_", " "),
                "confidence": confidence_percent,
                "suggestions": [],
                "low_confidence": True,
                "message": (
                    "The AI is not confident about this image (below 65%). "
                    "Please upload a clearer close-up of the affected leaves, "
                    "or consult a nearby agriculture officer/engineer."
                ),
            })

        # Get recommendation when confidence is acceptable
        rec = fertilizer_recommendations.get(
            disease,
            {
                "fertilizer": "General NPK fertilizer",
                "treatment": "Consult local agricultural expert"
            }
        )

        return jsonify({
            "disease": disease.replace("_", " "),
            "confidence": confidence_percent,
            "suggestions": [rec["fertilizer"], rec["treatment"]],
            "low_confidence": False,
        })

    except Exception as e:

        tb = traceback.format_exc()
        print("Prediction Error:", e)
        print(tb)

        payload = {"error": str(e)}
        if app.debug:
            payload["traceback"] = tb

        return jsonify(payload), 500


@app.route('/api/ask-disease', methods=['POST'])
def ask_disease():
    """Use Gemini Flash 2.5 to answer questions about a detected disease."""
    payload = request.get_json(silent=True) or {}

    question = payload.get("question", "").strip()
    disease = payload.get("disease", "").strip()
    suggestions = payload.get("suggestions") or []
    language = (payload.get("language") or "en").strip().lower()
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not disease:
        return jsonify({"error": "Disease name is required"}), 400

    if not isinstance(suggestions, list):
        suggestions = []

    client = _build_gemini_client()
    if client is None:
        return jsonify({
            "error": "Gemini API key is not configured on the server. "
                     "Set GEMINI_API_KEY environment variable and restart the app."
        }), 500

    context = _build_disease_context(disease, suggestions)

    try:
        if language == "te":
            system_text = (
                "You are an agricultural assistant helping farmers in India. "
                "Answer ONLY in clear, simple TELUGU (te-IN) that rural farmers can understand. "
                "Do NOT include any English sentences. Use English only for medicine/product names if needed."
            )
        else:
            system_text = (
                "You are an agricultural assistant helping farmers understand plant diseases "
                "and treatments in very simple English. Avoid technical jargon."
            )

        contents = [
            Content(
                parts=[
                    Part(
                        system_text
                    )
                ]
            ),
            Content(
                role="user",
                parts=[
                    Part(
                        (
                            "Answer the farmer's question in TELUGU only.\n\n"
                            if language == "te"
                            else "Answer the farmer's question in simple English.\n\n"
                        )
                        + f"Context about the detected disease: {context}\n\n"
                        f"Farmer's question: {question}"
                    )
                ],
            ),
        ]

        # Try preferred model, then fall back if service is overloaded.
        preferred_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")
        try:
            result = _gemini_generate_with_retries(
                client, model=preferred_model, contents=contents
            )
        except Exception as e:  # noqa: BLE001
            if preferred_model != fallback_model and _looks_like_overloaded_gemini_error(e):
                result = _gemini_generate_with_retries(
                    client, model=fallback_model, contents=contents
                )
            else:
                raise

        answer = getattr(result, "text", "").strip()
        if not answer and getattr(result, "candidates", None):
            # Fallback for older client response shapes
            first = result.candidates[0]
            if first and getattr(first, "content", None):
                answer = "".join(
                    part.text for part in first.content.parts if getattr(part, "text", None)
                ).strip()

        if not answer:
            return jsonify({"error": "Gemini did not return any text response."}), 500

        return jsonify({"answer": answer})

    except Exception as e:  # noqa: BLE001
        print("Gemini Error:", e)
        if _looks_like_overloaded_gemini_error(e):
            return jsonify({
                "error": "Gemini is temporarily overloaded (503). Please try again in 10–20 seconds."
            }), 503
        return jsonify({"error": f"Gemini API error: {e}"}), 500

# -------------------------------
# Run Server
# -------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=8000,
        use_reloader=False
    )