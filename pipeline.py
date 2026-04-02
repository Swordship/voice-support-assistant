import json
import os
import io
import time
from datetime import date
from pathlib import Path

import edge_tts
from groq import Groq, BadRequestError, InternalServerError
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    orders = json.loads((DATA_DIR / "orders.json").read_text())
    policies = json.loads((DATA_DIR / "policies.json").read_text())
    return orders, policies


# ----------------------------
# STEP 1: Audio → Text
# ----------------------------

def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe audio using Groq Whisper.
    Raises ValueError if audio is empty, silent, or unintelligible.
    """
    if not audio_bytes:
        raise ValueError("Audio file is empty.")

    try:
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3",
            language="en",
        )
    except BadRequestError as e:
        raise ValueError(f"Audio could not be processed: {e}")

    transcript = transcription.text.strip()

    if not transcript:
        raise ValueError("No speech detected in the audio.")

    # Whisper sometimes returns filler for silent audio
    FILLER_PHRASES = {"you", "thank you", "thanks", ".", "...", "the"}
    if transcript.lower() in FILLER_PHRASES:
        raise ValueError("No meaningful speech detected in the audio.")

    return transcript


# ----------------------------
# STEP 2: Text → LLM Response
# ----------------------------

SYSTEM_PROMPT = """You are a friendly customer support assistant for ShopEase, an e-commerce platform.
Today's date is {today}.

Use ONLY the data below to answer customer queries. Do not make up order details.

ORDERS:
{orders}

POLICIES:
{policies}

Rules:
- Be brief and conversational (2-3 sentences max).
- For order status: find the order by order ID or item name and report clearly.
- For return requests: check if delivery_date + return_window_days >= today.
  If still within window: confirm eligibility and state conditions.
  If expired: apologise clearly and say the return window has closed.
- For refund/policy questions: use the policies data.
- If the order is in_transit or processing: explain it cannot be returned yet.
- If the order ID or item is not found: apologise and suggest contacting support@shopease.com.
- If the query is completely unrelated to orders, returns, or refunds: politely say you can only help with order and return queries.
- Never expose user_id to the customer.
- If the query is vague (no order ID or item name mentioned): ask one clarifying question only.
"""


def generate_response(query: str, orders: list, policies: dict, retries: int = 3) -> str:
    """
    Generate support response using Groq Llama.
    Retries up to 3 times on rate limit or server errors.
    """
    prompt = SYSTEM_PROMPT.format(
        today=date.today().isoformat(),
        orders=json.dumps(orders, indent=2),
        policies=json.dumps(policies, indent=2),
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()

        except InternalServerError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
                continue
            raise

    raise RuntimeError("LLM service unavailable after multiple retries.")


# ----------------------------
# STEP 3: Text → Audio
# ----------------------------

async def synthesize(text: str) -> bytes:
    """Convert text to natural speech using Edge TTS."""
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    audio_buffer.seek(0)
    return audio_buffer.read()


# ----------------------------
# Full pipeline
# ----------------------------

async def run_pipeline(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """
    End-to-end pipeline.
    Returns transcript, response text, and audio bytes.
    Raises ValueError for bad input, RuntimeError for service failures.
    """
    orders, policies = load_data()

    transcript = transcribe(audio_bytes, filename)
    response_text = generate_response(transcript, orders, policies)
    audio_response = await synthesize(response_text)

    return {
        "transcript": transcript,
        "response": response_text,
        "audio_bytes": audio_response,
    }