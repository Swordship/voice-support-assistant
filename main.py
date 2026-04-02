import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pipeline import run_pipeline
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
load_dotenv()
import json
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

app = FastAPI(
    title="Voice AI Support Assistant",
    description="E-commerce support — voice in, text + voice out.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(
    audio: UploadFile = File(...),
    history: str = Form(default="[]")
):
    SUPPORTED = {"wav", "mp3", "m4a", "webm", "ogg", "flac", "mp4", "mpeg"}
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename and "." in audio.filename else ""

    if ext not in SUPPORTED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '.{ext}'. Supported: {', '.join(SUPPORTED)}",
        )

    audio_bytes = await audio.read()

    try:
        conversation_history = json.loads(history)
    except Exception:
        conversation_history = []

    try:
        result = await run_pipeline(audio_bytes, audio.filename, conversation_history)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    return JSONResponse(content={
        "transcript": result["transcript"],
        "response": result["response"],
        "audio_base64": base64.b64encode(result["audio_bytes"]).decode(),
        "audio_format": "mp3",
        "history": result["history"],
        "latency": result["latency"],
    })
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)