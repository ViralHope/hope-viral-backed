from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import shutil
import os
from pydantic import BaseModel

app = FastAPI()

# Definicja modelu odpowiedzi
class AnalysisResult(BaseModel):
    loop_score: float
    is_seamless_loop: bool
    recommendation: str

def check_video_loop(video_path, threshold=15.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return None
    
    # Odczyt pierwszej klatki
    success, first_frame = cap.read()
    
    # Przejście do ostatniej klatki
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    success, last_frame = cap.read()
    cap.release()
    
    if not success:
        return None
    
    # Obliczenie błędu MSE
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    mse = np.mean((gray_first.astype("float") - gray_last.astype("float")) ** 2)
    
    loop_score = max(0, min(100, 100 - (mse * 1.5)))
    is_smooth = mse < threshold
    
    return {
        "loop_score": round(loop_score, 2),
        "is_seamless_loop": is_smooth,
        "recommendation": "Pętla jest idealna." if is_smooth else "Zmień montaż końcowy."
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_video(file: UploadFile = File(...)):
    # Zapis tymczasowy
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = check_video_loop(temp_path)
        if not result:
            raise HTTPException(status_code=400, detail="Nie udało się przeanalizować wideo.")
        return result
    finally:
        # Sprzątanie
        if os.path.exists(temp_path):
            os.remove(temp_path)
import os
import uvicorn

if __name__ == "__main__":
    # Render automatycznie przypisze port do zmiennej środowiskowej PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

