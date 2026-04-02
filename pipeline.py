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


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
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

    FILLER_PHRASES = {"you", "thank you", "thanks", ".", "...", "the"}
    if transcript.lower() in FILLER_PHRASES:
        raise ValueError("No meaningful speech detected in the audio.")

    return transcript


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
- If order ID or item is not found: apologise and suggest contacting support@shopease.com.
- NEVER list all orders in the database. If asked, say you can only look up specific orders by order ID or item name.
- NEVER expose user_id or internal database fields to the customer.
- If the query is completely unrelated to orders, returns, or refunds: politely say you can only help with order and return queries.
- If the query is vague (no order ID or item name mentioned): ask one clarifying question only.
- Only share details of the specific order the customer is asking about, never share other customers orders.
"""


def generate_response(
    query: str,
    orders: list,
    policies: dict,
    history: list = [],
    retries: int = 3
) -> str:
    prompt = SYSTEM_PROMPT.format(
        today=date.today().isoformat(),
        orders=json.dumps(orders, indent=2),
        policies=json.dumps(policies, indent=2),
    )

    messages = [{"role": "system", "content": prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()

        except InternalServerError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError("LLM service unavailable after multiple retries.")


async def synthesize(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    audio_buffer.seek(0)
    return audio_buffer.read()


async def run_pipeline(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    history: list = []
) -> dict:
    orders, policies = load_data()

    t0 = time.perf_counter()
    transcript = transcribe(audio_bytes, filename)
    stt_ms = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    response_text = generate_response(transcript, orders, policies, history)
    llm_ms = round((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    audio_response = await synthesize(response_text)
    tts_ms = round((time.perf_counter() - t2) * 1000)

    # Append this turn to history
    updated_history = history + [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": response_text},
    ]

    return {
        "transcript": transcript,
        "response": response_text,
        "audio_bytes": audio_response,
        "history": updated_history,
        "latency": {
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": stt_ms + llm_ms + tts_ms,
        }
    }