from pathlib import Path
import os




# Paths
ROOT = Path(__file__).resolve().parent.parent   # Go 2 folders up from this Python file to get the project root folder means our main deep-research folder
PAPERS_DIR = ROOT / "papers"                    # Automatic detect path of papers folder
DATA_DIR = ROOT / "data"                        # Automatic detect path of data folder
QDRANT_PATH = str(DATA_DIR / "qdrant")
MANIFEST_PATH = DATA_DIR / "manifest.json"

os.environ.setdefault("FASTEMBED_CACHE_PATH", str(ROOT / ".models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Models
DENSE_MODEL = "BAAI/bge-base-en-v1.5"        # Embedding model. Token To Vector Transformation
DENSE_DIM = 768                              # Every vector from this model is exactly 768 numbers
SPARSE_MODEL = "Qdrant/bm25"                 # Retrieval Model (keyword search)
RERANK_MODEL = "BAAI/bge-reranker-base"


# LLM Models
LLM_MODEL = "qwen3:8b"
NUM_CTX = 8192                               # Context window available to Qwen for question + retrieved evidence + answer

# Chunking
CHUNK_SIZE = 800                             # Controls how large each piece of the document should be.
CHUNK_OVERLAP = 100                          # Amount of information repeated between neighboring chunks.

# Retrieval Settings
COLLECTION = "papers"                        # Name of the Qdrant collection where your research-paper vectors/data are stored.
CANDIDATES = 25                              # Number of possible relevant chunks you initially retrieve before final filtering/reranking.
RRF_K = 60                                   # It controls how strongly ranking positions affect the fusion.
TOP_K = 5                                    # Final number of highest-ranked chunks you keep/send onward.
USE_RERANKER = False








