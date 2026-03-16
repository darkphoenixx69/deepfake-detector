from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from PIL import Image
import io
import tempfile
import os

app = FastAPI(title="Deepfake Detector API")

# Allow requests from your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your Vercel URL in production e.g. ["https://your-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──────────────────────────────────────────────

print("Loading image deepfake detector...")
image_detector = pipeline(
    "image-classification",
    model="Organika/sdxl-detector",   # Detects AI-generated images
    device=0 if torch.cuda.is_available() else -1,
)

print("Loading text AI detector...")
text_tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
text_model = AutoModelForSequenceClassification.from_pretrained(
    "Hello-SimpleAI/chatgpt-detector-roberta"
)
text_model.eval()

print("All models loaded ✓")


# ── Helper ────────────────────────────────────────────────────────────────────

def interpret_image_result(results: list) -> dict:
    """Turn raw HF output into a clean verdict."""
    label_map = {}
    for r in results:
        label_map[r["label"].lower()] = r["score"]

    # Model labels: "artificial" vs "real"  (varies by model)
    ai_score = label_map.get("artificial", label_map.get("fake", label_map.get("ai", 0)))
    real_score = label_map.get("real", label_map.get("human", 1 - ai_score))

    confidence = round(max(ai_score, real_score) * 100, 1)
    is_fake = ai_score > real_score

    return {
        "verdict": "AI-Generated / Fake" if is_fake else "Likely Real",
        "confidence": confidence,
        "ai_score": round(ai_score * 100, 1),
        "real_score": round(real_score * 100, 1),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Deepfake Detector API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Accepts an image file and returns a deepfake verdict."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp…)")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="Image too large. Max 10 MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = image_detector(image)
        verdict = interpret_image_result(results)
        return {"type": "image", **verdict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Samples frames from a video and runs the image detector on each.
    Returns an aggregated verdict.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video (mp4, mov…)")

    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:  # 100 MB limit
        raise HTTPException(status_code=413, detail="Video too large. Max 100 MB.")

    try:
        import cv2
        import numpy as np

        # Write to a temp file so OpenCV can read it
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = min(10, total_frames)  # sample up to 10 frames
        interval = max(1, total_frames // sample_count)

        ai_scores = []
        for i in range(sample_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            results = image_detector(pil_img)
            verdict = interpret_image_result(results)
            ai_scores.append(verdict["ai_score"])

        cap.release()
        os.unlink(tmp_path)

        avg_ai = round(sum(ai_scores) / len(ai_scores), 1) if ai_scores else 0
        avg_real = round(100 - avg_ai, 1)
        is_fake = avg_ai > 50

        return {
            "type": "video",
            "verdict": "AI-Generated / Deepfake" if is_fake else "Likely Real",
            "confidence": round(max(avg_ai, avg_real), 1),
            "ai_score": avg_ai,
            "real_score": avg_real,
            "frames_analyzed": len(ai_scores),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


@app.post("/detect/text")
async def detect_text(payload: dict):
    """
    Accepts { "text": "..." } and returns whether the text is AI-generated.
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text too short — provide at least 50 characters.")
    if len(text) > 5000:
        text = text[:5000]  # truncate silently

    try:
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = text_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        # Label 0 = Human, Label 1 = AI (ChatGPT)
        human_score = round(probs[0] * 100, 1)
        ai_score = round(probs[1] * 100, 1)
        is_fake = ai_score > human_score

        return {
            "type": "text",
            "verdict": "Likely AI-Generated" if is_fake else "Likely Human-Written",
            "confidence": round(max(human_score, ai_score), 1),
            "ai_score": ai_score,
            "real_score": human_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
