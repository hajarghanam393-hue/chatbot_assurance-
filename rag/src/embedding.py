from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

        self.md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section")],
            strip_headers=False
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _chunk_md_documents(self, documents: List[Any]) -> List[Any]:
        """Chunk MD documents by section (##), puis re-split si une section depasse chunk_size."""
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking MD"):
            sections = self.md_header_splitter.split_text(doc.page_content)
            for section in sections:
                section.metadata = {**doc.metadata, **section.metadata}

            sub_chunks = self.recursive_splitter.split_documents(sections)

            for idx, chunk in enumerate(sub_chunks):
                chunk.metadata["chunk_index"] = idx

            all_chunks.extend(sub_chunks)

        print(f"[INFO] Split {len(documents)} MD documents into {len(all_chunks)} chunks.")
        return all_chunks

    def _chunk_csv_documents(self, documents: List[Any]) -> List[Any]:
        """
        CSVLoader produit deja 1 doc = 1 ligne. On ne chunk que si une ligne
        depasse chunk_size, pour eviter de casser la structure colonne:valeur.
        """
        chunks = []
        for doc in tqdm(documents, desc="Chunking CSV"):
            if len(doc.page_content) <= self.chunk_size:
                sub_chunks = [doc]
            else:
                sub_chunks = self.recursive_splitter.split_documents([doc])

            for idx, chunk in enumerate(sub_chunks):
                chunk.metadata["chunk_index"] = idx

            chunks.extend(sub_chunks)

        print(f"[INFO] Processed {len(documents)} CSV documents into {len(chunks)} chunks.")
        return chunks

    def chunk_documents(self, documents: List[Any], doc_type: str = "generic") -> List[Any]:
        if doc_type == "md":
            chunks = self._chunk_md_documents(documents)
        elif doc_type == "csv":
            chunks = self._chunk_csv_documents(documents)
        else:
            chunks = self.recursive_splitter.split_documents(documents)
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
            print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")

        return chunks

    def embed_chunks(self, chunks: List[Any], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Utilise au moment de la recherche, pas de l'indexation."""
        return self.model.encode(
            query,
            normalize_embeddings=normalize,
        )
