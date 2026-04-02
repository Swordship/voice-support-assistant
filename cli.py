import asyncio
import sys
from pathlib import Path
from pipeline import run_pipeline


async def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py <audio_file>")
        print("Example: python cli.py test_query.wav")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Error: file not found — {audio_path}")
        sys.exit(1)

    print(f"\n🎤  Processing: {audio_path.name}")
    print("─" * 40)

    result = await run_pipeline(audio_path.read_bytes(), audio_path.name)

    print(f"📝  You said  : {result['transcript']}")
    print(f"💬  Response  : {result['response']}")

    output_path = audio_path.parent / f"{audio_path.stem}_response.mp3"
    output_path.write_bytes(result["audio_bytes"])
    print(f"🔊  Audio saved: {output_path}")
    print("─" * 40)


if __name__ == "__main__":
    asyncio.run(main())