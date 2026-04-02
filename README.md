# 🎙️ Voice AI Support Assistant

A voice-enabled customer support system for e-commerce returns and orders.  
Speak your query — get a text response and a spoken audio reply instantly.

---

## Demo

```
You  : "I want to return my wireless headphones, order ORD123"
Bot  : "Your Wireless Headphones are within the 7-day return window!
        Please ensure the item is unused and in its original packaging."

You  : "What about the refund?"        ← no context needed, remembers ORD123
Bot  : "Refunds are processed to your original payment method within 5 business days."
```

---

## Pipeline

```
your voice query (.wav / .mp3 / .webm)
         │
         ▼
  Groq Whisper Large v3     ← speech to text
         │
         ▼
  Groq Llama 3.1 8B         ← understands query + conversation history
  + orders.json             ← looks up order data
  + policies.json           ← reads return / refund policies
         │
         ▼
  Microsoft Edge TTS        ← natural voice response
  (en-US-JennyNeural)
         │
         ▼
  JSON response
  { transcript, response, audio_base64, history, latency }
```

---

## Features

- 🎤 **Voice input** — upload audio file or record directly in browser
- 💬 **Multi-turn memory** — remembers context across the conversation
- 🔊 **Natural voice output** — Microsoft Edge TTS, not robotic
- ⏱️ **Latency breakdown** — STT / LLM / TTS / total per request
- 🛡️ **Robust error handling** — empty audio, silent input, unsupported format, service retries
- 🌐 **Browser UI** — clean chat interface with live mic recording
- ⌨️ **CLI** — terminal interface for quick testing

---

## Stack

| Layer | Service | Why |
|---|---|---|
| Speech → Text | Groq Whisper Large v3 | Fast, accurate, free tier |
| LLM | Groq Llama 3.1 8B Instant | Low latency, free tier |
| Text → Speech | Edge TTS (en-US-JennyNeural) | Natural voice, completely free |
| API | FastAPI + Uvicorn | Async, auto Swagger docs |

---

## Setup

### 1. Prerequisites
- Python 3.11+
- Free Groq API key → [console.groq.com](https://console.groq.com)

### 2. Clone and install

```bash
git clone <repo-url>
cd voice-support-assistant

python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and add:
```
GROQ_API_KEY=your-key-here
```

---

## Running

### API + Browser UI

```bash
uvicorn main:app --reload
```

- Browser UI → **http://localhost:8000**
- Swagger docs → **http://localhost:8000/docs**
- Health check → **http://localhost:8000/health**

### CLI

```bash
python cli.py your_query.wav
```

Output:
```
🎤  Processing: your_query.wav
────────────────────────────────────────
📝  You said  : I want to return my wireless headphones
💬  Response  : Your Wireless Headphones are within the return window...
🔊  Audio saved: your_query_response.mp3
────────────────────────────────────────
```

### Quick pipeline test

```bash
python test_pipeline.py
```

---

## API Reference

### `POST /query`

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| audio | file | ✅ | Audio file (wav, mp3, m4a, webm, ogg, flac) |
| history | string | ❌ | JSON array of previous turns (default: `[]`) |

**Response:** `application/json`

```json
{
  "transcript": "Where is my order ORD124?",
  "response": "Your Running Shoes are currently in transit, expected by March 30th!",
  "audio_base64": "<base64-encoded MP3>",
  "audio_format": "mp3",
  "history": [
    { "role": "user", "content": "Where is my order ORD124?" },
    { "role": "assistant", "content": "Your Running Shoes are currently in transit..." }
  ],
  "latency": {
    "stt_ms": 820,
    "llm_ms": 1100,
    "tts_ms": 310,
    "total_ms": 2230
  }
}
```

**Error codes:**

| Code | Reason |
|---|---|
| 400 | Empty audio or no speech detected |
| 415 | Unsupported audio format |
| 503 | LLM service unavailable after retries |
| 500 | Unexpected server error |

### `GET /health`

```json
{ "status": "ok" }
```

---

## Project Structure

```
voice-support-assistant/
├── main.py              # FastAPI server
├── pipeline.py          # STT → LLM → TTS logic
├── cli.py               # CLI interface
├── test_pipeline.py     # Quick pipeline test
├── static/
│   └── index.html       # Browser UI
├── data/
│   ├── orders.json
│   └── policies.json
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dataset

Located in `data/`. The LLM reads this directly as source of truth.

**orders.json** — 4 orders (matches provided problem statement + 1 extra)
**policies.json** — return window, refund method, support hours

Queries the assistant handles:
- **Order status** → looks up by order ID or item name
- **Return eligibility** → checks `delivery_date + return_window_days` vs today
- **Refund / policy questions** → reads from policies.json
- **Vague queries** → asks one clarifying question
- **Out of scope** → politely declines, stays on topic

---

## Assumptions

1. **No authentication** — all orders are accessible without login. In production, queries would be scoped to an authenticated user session so users can only see their own orders.

2. **Order identification** — users are expected to mention an order ID (e.g. ORD123) or item name. The assistant asks a clarifying question if neither is provided.

3. **Return window logic delegated to LLM** — the system prompt includes today's date and full order data. The LLM reasons about eligibility. As of submission date: ORD123 (Wireless Headphones, delivered 2026-03-20) is within its 7-day window. ORD125 (Smart Watch, delivered 2026-03-10) window has expired.

4. **ORD124 expected delivery** — the problem statement shows `2026-03-30` which is already past. We keep the original date and document it. The assistant correctly reports it as in-transit without a confirmed delivery.

5. **English only** — Whisper supports multilingual but the system prompt and responses are English only.

6. **Static dataset** — orders are loaded from JSON at startup. Production would replace this with a database query scoped to the authenticated user.

7. **Database privacy** — the assistant will never list all orders when asked. It only responds about the specific order a user asks about.

---

## Design Decisions & Tradeoffs

### Groq over OpenAI
OpenAI Whisper and GPT-4o-mini require billing setup. Groq provides both Whisper Large v3 and Llama 3.1 on a genuinely free tier with no credit card — practical for a take-home with no infrastructure budget, and fast enough for production latency targets.

### Prompt injection over RAG
The dataset is 4 orders and a small policy file — it fits in the context window with room to spare. Injecting it directly is simpler, faster, and easier to debug than a vector store. If the dataset grew to thousands of orders, the right upgrade is tool calling with a `lookup_order(order_id)` database function.

### Multi-turn conversation via history passing
Conversation history is maintained on the client side and sent with each request as a JSON array. This keeps the server stateless and horizontally scalable — no session store needed. The tradeoff is larger payloads as history grows, which can be mitigated with a sliding window (last N turns).

### Edge TTS over gTTS
gTTS works but sounds robotic. Microsoft Edge TTS (`edge-tts`) is free, requires no API key, and produces significantly more natural speech using neural voices. The tradeoff is it's an unofficial API — for production, ElevenLabs or OpenAI TTS would be more reliable.

### Async pipeline
FastAPI runs on an async event loop. Edge TTS is natively async. Making `synthesize()` and `run_pipeline()` async avoids event loop conflicts and keeps the server non-blocking under concurrent requests.

### Base64 audio in JSON
A single JSON response is simpler to consume than multipart. The tradeoff is ~33% size overhead from base64 encoding. For production with large audio, a presigned S3 URL would be more efficient.

---

## Potential Improvements

- **Streaming TTS** — stream audio chunks as they generate instead of waiting for the full file, cutting perceived latency significantly
- **Tool calling** — replace prompt-injected dataset with a `lookup_order(order_id)` function so the LLM fetches only what it needs, scales to large datasets
- **User authentication** — scope orders to a session so "my order" resolves without an explicit order ID
- **History sliding window** — cap history to last 10 turns to prevent token bloat on long conversations
- **Retry with backoff** — currently retries on `InternalServerError` only, could extend to rate limit errors with exponential backoff
- **Unit tests** — mock the Groq client and test pipeline functions in isolation
- **Deeper STT** — Deepgram Nova-2 is faster and supports real-time streaming for lower latency voice experiences