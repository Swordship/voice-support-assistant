import asyncio
from pathlib import Path
from pipeline import run_pipeline

async def main():
    audio = Path("test_query.wav").read_bytes()
    result = await run_pipeline(audio, "test_query.wav")

    print("Transcript :", result["transcript"])
    print("Response   :", result["response"])
    print()
    print("⏱️  STT      :", result["latency"]["stt_ms"], "ms")
    print("⏱️  LLM      :", result["latency"]["llm_ms"], "ms")
    print("⏱️  TTS      :", result["latency"]["tts_ms"], "ms")
    print("⏱️  Total    :", result["latency"]["total_ms"], "ms")

    Path("response.mp3").write_bytes(result["audio_bytes"])
    print("\nAudio saved: response.mp3")

asyncio.run(main())