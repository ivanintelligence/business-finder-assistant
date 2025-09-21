# ingest_database.py — streaming/batched CSV ingestion with locality priority + row cap (fixed)
from dotenv import load_dotenv; load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb, hashlib, os, glob, csv
import pandas as pd
import re
from typing import Optional

DATA_PATH    = "data"         # folder containing one or more CSVs
CHROMA_PATH  = "chroma_db"
COLLECTION   = "example_collection"
MANIFEST_CSV = "ingest_manifest.csv"

# ----------------- Tuning knobs -----------------
MAX_ROWS_TOTAL  = 100_000     # hard cap
ROWS_PER_CHUNK  = 5_000       # pandas chunk size
ADD_BATCH       = 512         # how many Documents per Chroma add
MAX_TEXT_CHARS  = 1_500       # truncate long row text for cheaper embeddings
COLUMNS: Optional[list[str]] = None  # e.g., ["name","address","locality"]; None=use all
LOCALITY_COL_CANDIDATES = ["locality", "city", "town", "municipality"]  # try in order
PRIORITY_LOCALITIES = [
    "new york", "nyc", "new york city"  # highest priority localities, case-insensitive
]
IGNORE_COLS = {"unnamed: 0", "index"}  # columns to ignore in text

# Optional: speedups / stability
CSV_READ_KW = {
    # "engine": "pyarrow",   # uncomment if installed; often faster
    "dtype": "string",
    "index_col": False,
    "low_memory": True,
    "on_bad_lines": "skip",
}

# ----------------- Embeddings & Vector DB -----------------
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    # model_kwargs={"device": "mps"}  # uncomment on Apple Silicon for GPU accel
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
vector_store  = Chroma(client=chroma_client, collection_name=COLLECTION, embedding_function=embeddings_model)

# ----------------- Helpers -----------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Optional: remove annoying unnamed columns early
    drop_these = [c for c in df.columns if c.startswith("unnamed:")]
    if drop_these:
        df.drop(columns=drop_these, inplace=True, errors="ignore")
    return df

def pick_locality_col(df: pd.DataFrame) -> Optional[str]:
    for c in LOCALITY_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def build_locality_mask(series: pd.Series) -> pd.Series:
    # Match any of the priority localities (case-insensitive, substring)
    patt = "|".join(re.escape(s) for s in PRIORITY_LOCALITIES)
    return series.str.contains(patt, case=False, na=False)

def row_to_text(row_dict: dict) -> str:
    keys = list(COLUMNS) if COLUMNS else list(row_dict.keys())
    keys = [k for k in keys if str(k).lower() not in IGNORE_COLS]
    parts = []
    for key in keys:
        val = row_dict.get(key, "")
        s = str(val).strip()
        if s and s.lower() != "nan":
            parts.append(f"{key}: {s}")
    text = "\n".join(parts)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + " ..."
    return text

def make_id(source_path: str, row_idx: int, content: str) -> str:
    return hashlib.sha1((source_path + "|" + str(row_idx) + "|" + content).encode("utf-8")).hexdigest()

def ensure_manifest_header():
    if not os.path.exists(MANIFEST_CSV):
        with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "row", "locality", "doc_id"])  # header

def append_manifest(rows):
    if not rows:
        return
    with open(MANIFEST_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

# ----------------- Core ingestion -----------------
def process_csv(path: str, global_state: dict):
    print(f"[Ingest] {path}")
    total_added = global_state["added"]
    chunk_idx = 0

    for df in pd.read_csv(path, chunksize=ROWS_PER_CHUNK, **CSV_READ_KW):
        if total_added >= MAX_ROWS_TOTAL:
            break

        df = normalize_cols(df)
        loc_col = pick_locality_col(df)

        if loc_col is None:
            # No locality column → everything is "remainder"
            pri_df = df.iloc[0:0]
            rem_df = df
        else:
            mask = build_locality_mask(df[loc_col])
            pri_df = df[mask]
            rem_df = df[~mask]

        # Process priority first, then remainder until cap is hit
        for sub_df, tag in [(pri_df, "priority"), (rem_df, "remainder")]:
            if sub_df.empty:
                continue

            # Preserve each row's original position within this chunk for a stable file-level index
            sub_df = sub_df.reset_index(drop=False)  # adds "index" column with original positions

            docs, ids, manifest_rows = [], [], []
            # IMPORTANT: iterate with name=None to get plain tuples, then zip → avoids attribute-name issues
            for row_vals in sub_df.itertuples(index=False, name=None):
                if total_added >= MAX_ROWS_TOTAL:
                    break

                row_dict = dict(zip(sub_df.columns, row_vals))
                orig_in_chunk = int(row_dict.get("index", 0))  # original index before filtering
                row_idx = chunk_idx * ROWS_PER_CHUNK + orig_in_chunk  # approx. absolute row in file

                content = row_to_text(row_dict)
                if not content:
                    continue

                locality_val = str(row_dict.get(loc_col, "")) if loc_col else ""
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": path,
                        "row": row_idx,
                        "locality": locality_val,
                        "priority_bucket": tag
                    },
                )
                doc_id = make_id(path, row_idx, content)

                docs.append(doc)
                ids.append(doc_id)
                manifest_rows.append([path, row_idx, locality_val, doc_id])

                if len(docs) >= ADD_BATCH:
                    vector_store.add_documents(docs, ids=ids)
                    append_manifest(manifest_rows)
                    total_added += len(docs)
                    print(f"  [+] {total_added}/{MAX_ROWS_TOTAL} (added {len(docs)} from {tag})")
                    docs, ids, manifest_rows = [], [], []

                    if total_added >= MAX_ROWS_TOTAL:
                        break

            # flush remainder
            if docs and total_added < MAX_ROWS_TOTAL:
                vector_store.add_documents(docs, ids=ids)
                append_manifest(manifest_rows)
                total_added += len(docs)
                print(f"  [+] {total_added}/{MAX_ROWS_TOTAL} (final flush {tag})")

            if total_added >= MAX_ROWS_TOTAL:
                break

        chunk_idx += 1

    global_state["added"] = total_added
    print(f"[Done] {path} → added so far: {total_added}")

def main():
    ensure_manifest_header()
    csv_paths = glob.glob(os.path.join(DATA_PATH, "**/*.csv"), recursive=True)
    if not csv_paths:
        print(f"No CSV files found under {DATA_PATH}")
        return

    state = {"added": 0}
    for path in csv_paths:
        if state["added"] >= MAX_ROWS_TOTAL:
            break
        process_csv(path, state)

    try:
        count = vector_store._collection.count()
        print(f"[Summary] Collection '{COLLECTION}' now has {count} items at {CHROMA_PATH}.")
        print(f"[Manifest] Wrote details to {MANIFEST_CSV}")
    except Exception:
        print("[Summary] Ingestion complete.")

if __name__ == "__main__":
    main()