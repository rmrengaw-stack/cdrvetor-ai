from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid

from vectorizer import vectorize_image

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CDRvetor AI")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    ext = Path(file.filename or "upload.png").suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        raise HTTPException(status_code=400, detail="Formato não suportado. Use PNG, JPG, WEBP ou BMP.")

    job_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}.svg"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        vectorize_image(str(input_path), str(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao vetorizar: {e}")

    return FileResponse(
        output_path,
        media_type="image/svg+xml",
        filename="cdrvetor-ai.svg"
    )
