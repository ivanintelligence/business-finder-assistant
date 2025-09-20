# chatbot.py
import os
import re
from dotenv import load_dotenv

# Quiet tokenizers warning before imports that might touch it
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import gradio as gr
from openai import OpenAI

# ----------------- Config -----------------
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
COLLECTION = "example_collection"

HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# ----------------- Embeddings & Vector DB -----------------
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# Persistent Chroma (chromadb>=0.5)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
vector_store = Chroma(
    client=chroma_client,
    collection_name=COLLECTION,
    embedding_function=embeddings_model,
)

# Retriever (keep simple; we’ll add our own score filter fallback)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# ----------------- LLM Client (HF Router) -----------------
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# ----------------- Prompts -----------------
INSTRUCTIONS_RAG = """
You are an assistant which answers questions based on knowledge provided to you.
While answering, do not use your internal knowledge; use only the text in the "Context".
Do not mention the Context or that it was provided.
If the answer is not in the Context, say you don't know.
Keep answers concise and accurate.
""".strip()

INSTRUCTIONS_CHAT = """
You are a friendly, concise assistant. For greetings or small talk, respond naturally and briefly.
If the user asks something not covered by any provided documents, answer normally without inventing facts.
""".strip()

SMALLTALK_RE = re.compile(
    r"^(hi|hello|hey|yo|sup|ok|okay|thanks|thank you|good (morning|afternoon|evening)|how are you\??)\W*$",
    re.IGNORECASE,
)

# ----------------- Helpers -----------------
def _append_history_as_messages(msgs, history):
    """Accepts messages-mode (list[dict]) or legacy tuple-mode (list[tuple])."""
    if not isinstance(history, list) or not history:
        return
    if isinstance(history[0], dict):
        for h in history[-6:]:
            role = h.get("role")
            if role in {"user", "assistant"}:
                msgs.append({"role": role, "content": h.get("content", "")})
        return
    for u, a in history[-3:]:
        msgs.append({"role": "user", "content": u or ""})
        msgs.append({"role": "assistant", "content": a or ""})

def _fetch_relevant_knowledge(query, max_chars=4000, min_score=0.25):
    """
    Return (knowledge_text, n_docs). Tries retriever first;
    then falls back to similarity_search_with_relevance_scores if available.
    """
    # Primary path (may already apply its own scoring)
    docs = retriever.invoke(query) or []

    # Fallback with explicit scores filtering (newer LC versions support this)
    if not docs:
        try:
            pairs = vector_store.similarity_search_with_relevance_scores(query, k=8)
            docs = [doc for doc, score in pairs if score is None or score >= min_score]
        except Exception:
            try:
                docs = vector_store.similarity_search(query, k=8)
            except Exception:
                docs = []

    text = "\n\n".join(
        d.page_content for d in docs if getattr(d, "page_content", None)
    )
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n...[truncated]..."
    return text, len(docs)

# ----------------- Gradio callback -----------------
def stream_response(message, history):
    msg = (message or "").strip()

    # Route: small talk → pure chat; otherwise try RAG
    is_smalltalk = bool(SMALLTALK_RE.match(msg))
    knowledge, nctx = ("", 0) if is_smalltalk else _fetch_relevant_knowledge(msg)

    print(f"[Router] smalltalk={is_smalltalk}  ctx_docs={nctx}")

    if is_smalltalk:
        system_msg = INSTRUCTIONS_CHAT
        user_msg = msg
    elif nctx > 0:
        system_msg = INSTRUCTIONS_RAG
        user_msg = f"Question: {msg}\n\nContext:\n{knowledge}"
    else:
        # No relevant context → normal chat (keeps the bot conversational)
        system_msg = INSTRUCTIONS_CHAT
        user_msg = msg

    messages = [{"role": "system", "content": system_msg}]
    _append_history_as_messages(messages, history)
    messages.append({"role": "user", "content": user_msg})

    try:
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=messages,
            temperature=0.2,
            stream=True,
        )

        buffer = ""
        for chunk in stream:
            # Some events can be empty/keep-alive
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