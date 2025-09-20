# chatbot.py — Hybrid general chat + RAG-only-when-relevant
import os, re
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import gradio as gr
from openai import OpenAI

# ----------------- Config -----------------
CHROMA_PATH = "chroma_db"
COLLECTION = "example_collection"
HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# ----------------- Embeddings & Vector DB -----------------
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
vector_store = Chroma(
    client=chroma_client,
    collection_name=COLLECTION,
    embedding_function=embeddings_model,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# ----------------- LLM Client -----------------
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)

# ----------------- Prompts -----------------
INSTRUCTIONS_HYBRID = """
You are a helpful, concise assistant.

If a "Context" section is provided below, you MUST answer using only that Context.
- Do not add facts that are not supported by the Context.
- If the Context does not contain the answer, say you don't know based on the provided context.
- Do not mention that you used a Context.

If no Context is provided, answer normally like a general-purpose assistant.
Keep replies brief and clear.
""".strip()

SMALLTALK_RE = re.compile(
    r"^(hi|hello|hey|yo|sup|ok|okay|thanks|thank you|good (morning|afternoon|evening)|how are you\??)\W*$",
    re.IGNORECASE,
)

# ----------------- Helpers -----------------
def _append_history_as_messages(msgs, history):
    if not isinstance(history, list) or not history:
        return
    if isinstance(history[0], dict):  # messages-mode
        for h in history[-6:]:
            if h.get("role") in {"user", "assistant"}:
                msgs.append({"role": h["role"], "content": h.get("content", "")})
        return
    for u, a in history[-3:]:  # tuple-mode fallback
        msgs.append({"role": "user", "content": u or ""})
        msgs.append({"role": "assistant", "content": a or ""})

def _fetch_context(query: str, k: int = 8, min_score: float = 0.30, max_chars: int = 6000):
    """Return (context_text, n_docs). Uses scores when available; filters weak hits."""
    # Try scored search first
    try:
        pairs = vector_store.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, score in pairs if (score is None) or (score >= min_score)]
    except Exception:
        # Fallback to basic retriever
        docs = retriever.invoke(query) or []

    text = "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[truncated]..."
    return text, len(docs)

# ----------------- Gradio callback -----------------
def stream_response(message, history):
    query = (message or "").strip()

    # Small talk → skip retrieval entirely
    use_smalltalk = bool(SMALLTALK_RE.match(query))
    context, nctx = ("", 0) if use_smalltalk else _fetch_context(query)

    print(f"[Router] smalltalk={use_smalltalk}  ctx_docs={nctx}")

    # Build messages (single hybrid instruction)
    messages = [{"role": "system", "content": INSTRUCTIONS_HYBRID}]
    _append_history_as_messages(messages, history)

    user_content = f"Question: {query}"
    if nctx > 0:  # attach Context only when relevant
        user_content += f"\n\nContext:\n{context}"
    messages.append({"role": "user", "content": user_content})

    try:
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=messages,
            temperature=0.3,
            stream=True,
        )

        buffer = ""
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                buffer += text
                yield buffer
            if getattr(choice, "finish_reason", None):
                break

    except Exception as e:
        print("LLM error:", repr(e))
        yield "Sorry—something went wrong while streaming the model’s response. Please try again."

# ----------------- UI -----------------
demo = gr.ChatInterface(
    fn=stream_response,
    chatbot=gr.Chatbot(type="messages"),
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7,
    ),
)
demo.launch(share=False)