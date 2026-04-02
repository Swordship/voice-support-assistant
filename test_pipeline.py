import asyncio
from pathlib import Path
from pipeline import run_pipeline

async def main():
    audio = Path("test_query.wav").read_bytes()
    result = await run_pipeline(audio, "test_query.wav")
    print("Transcript :", result["transcript"])
    print("Response   :", result["response"])
    Path("response.mp3").write_bytes(result["audio_bytes"])
    print("Audio saved: response.mp3")

asyncio.run(main())