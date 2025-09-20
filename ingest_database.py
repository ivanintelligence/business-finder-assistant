from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb
import hashlib

load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
COLLECTION = "example_collection"

# Embeddings (normalize helps cosine search)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ✅ Persistent client (this is what actually writes to disk)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Vector store backed by persistent client
vector_store = Chroma(
    client=chroma_client,
    collection_name=COLLECTION,
    embedding_function=embeddings_model,
)

# Load PDFs
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = [d for d in loader.load() if d.page_content and d.page_content.strip()]

# Split into chunks (bigger chunks improve recall)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=180,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)

# Deterministic IDs (prevents duplicates on re-run)
def make_id(doc):
    src = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    key = (src + "|" + page + "|" + doc.page_content).encode("utf-8")
    return hashlib.sha1(key).hexdigest()

ids = [make_id(doc) for doc in chunks]

# Add (re-adding same IDs overwrites in Chroma)
vector_store.add_documents(documents=chunks, ids=ids)

# Sanity check
count = vector_store._collection.count()  # for quick debug
print(f"Ingested {len(chunks)} chunks; collection now has {count} items at {CHROMA_PATH}.")