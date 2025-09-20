from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
import os
from openai import OpenAI

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

def stream_response(message, history):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet the warning

    # --- RAG: retrieve relevant chunks ---
    docs = retriever.invoke(message)
    knowledge = "\n\n".join(doc.page_content for doc in docs) if docs else "No relevant docs."

    # --- Compose the prompt you already had ---
    rag_prompt = f"""
    You are an assistant which answers questions based on knowledge provided to you.
    While answering, you don't use your internal knowledge,
    but solely the information in the "The knowledge" section.
    You don't mention anything to the user about the provided knowledge.

    The question: {message}

    Conversation history: {history}

    The knowledge: {knowledge}
    """.strip()

    # OpenAI-compatible messages
    messages = [{"role": "user", "content": rag_prompt}]

    try:
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=messages,
            temperature=0.3,
            stream=True,
        )

        buffer = ""
        for chunk in stream:
            # Some events have no choices (keep-alives, etc.)
            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]

            # Pull text if present
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                buffer += text
                yield buffer  # incremental updates for Gradio

            # End-of-stream signal
            if getattr(choice, "finish_reason", None):
                break

    except Exception as e:
        print("LLM error:", repr(e))
        # Don't raise; yield a friendly message so Gradio doesn't show red "error"
        yield "Sorry—something went wrong while streaming the model’s response. Please try again."

# initiate the Gradio app
demo = gr.ChatInterface(
    fn=stream_response,
    chatbot=gr.Chatbot(type="messages"),          # fixes the deprecation
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7,
    ),
)

# Set share=True if you want a public link
demo.launch(share=False)