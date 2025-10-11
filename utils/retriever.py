import logging
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pyserini.search.lucene import LuceneSearcher
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pydantic import PrivateAttr
from utils.config import Config
from typing import List, Any, Optional
from tqdm import tqdm
import subprocess
import os
import json

logger = logging.getLogger(__name__)
config = Config()
class PyseriniBM25Retriever(BaseRetriever):
    
    index_dir: str
    k: int = 5
    k1: float = 0.9
    b: float = 0.4
    _searcher: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)

        s = LuceneSearcher(self.index_dir)
        s.set_bm25(self.k1, self.b)
        self._searcher = s

        logger.info(
            "PyseriniBM25Retriever ready: index=%s, k=%d, bm25(k1=%.3f, b=%.3f)",
            self.index_dir, self.k, self.k1, self.b
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        hits = self._searcher.search(query, k=self.k)
        docs: List[Document] = []
        for h in hits:
            raw = self._searcher.doc(h.docid).raw() or ""
            text, title = raw, None
            try:
                obj = json.loads(raw)
                text = obj.get("contents") or obj.get("text") or raw
                title = obj.get("title")
            except Exception:
                pass
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "docid": h.docid,
                        "score": float(h.score),
                        "title": title,
                    },
                )
            )
        return docs

def preprocess_corpus(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            doc = {
                "id": f"{i}",  # Changed to match qrels format
                "contents": line.strip()
            }
            with open(os.path.join(output_dir, f"doc{i}.json"), 'w') as out:
                json.dump(doc, out)


def build_index(input_dir, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    subprocess.run(cmd, check=True)

def create_index(cname):

    base_dir = f"data/{cname}"

    # Paths to the raw corpus, queries, and relevance label files
    corpus_file = os.path.join(base_dir, f"{cname}.dat")
    # processed_corpus_dir = os.path.join(base_dir, "corpus")

    # Directories where the processed corpus and index will be stored for toolkit
    processed_corpus_dir = f"processed_corpus/{cname}"
    os.makedirs(processed_corpus_dir, exist_ok=True)
    index_dir = f"indexes/{cname}"

    # Preprocess corpus
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}. Skipping preprocessing.")

    # Build index
    build_index(processed_corpus_dir, index_dir)
    return index_dir

def create_retriever():
    try:
        index_dir = create_index(config.PYSERINI_CNAME)
        base_retriever = PyseriniBM25Retriever(
            index_dir=index_dir,
            k=config.RETRIEVER_K,
            k1=config.PYSERINI_K1,
            b=config.PYSERINI_B,
        )
        logger.info(f"Created Pyserini base retriever on {index_dir}")

        # Set up LLM for contextual compression
        llm = config.get_llm()  # You'll need to implement this method in your Config class
        compressor = LLMChainExtractor.from_llm(llm)

        # Create contextual compression retriever
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

        logger.info("Retriever created successfully")
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}")
        raise
