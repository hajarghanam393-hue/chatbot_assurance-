import os
import math
import chromadb
import numpy as np
from typing import List, Any
import time


class VectorStore:
    '''Manage document embeddings in a ChromaDB vector store'''

    def __init__(self, collection_name: str = 'assurance_documents', persist_directory: str = None):
        '''
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        '''
        self.collection_name = collection_name
        # Chemin absolu basé sur l'emplacement du fichier, pas sur le cwd d'exécution
        self.persist_directory = persist_directory or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'vector_store'
        )
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        '''Initialize ChromaDB client and collection'''
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={'description': 'AssurAuto Maroc document embeddings for RAG', 'hnsw:space': 'cosine'}
            )
            print(f'[INFO] Vector store initialized. Collection: {self.collection_name}')
            print(f'[INFO] Existing documents in collection: {self.collection.count()}')
        except Exception as e:
            print(f'[ERROR] Error initializing vector store: {e}')
            raise

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        '''ChromaDB n'accepte que str/int/float/bool dans les métadonnées, pas None/NaN/listes.'''
        clean = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, float) and math.isnan(value):
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean
    
    @staticmethod
    def _build_doc_id(doc, fallback_index: int) -> str:
        '''
        ID stable basé sur l'ID métier du document (MD: document_id, CSV: record_id)
        + chunk_index, pour que upsert écrase correctement au lieu de dupliquer.
        '''
        base_id = doc.metadata.get('document_id') or doc.metadata.get('record_id')
        chunk_index = doc.metadata.get('chunk_index', fallback_index)

        if not base_id:
            # Fallback si aucun ID métier trouvé (ne devrait pas arriver en usage normal)
            source = doc.metadata.get('source', 'unknown')
            base_id = source

        return f"{base_id}::{chunk_index}"

    def add_documents(self, documents: List[Any], embeddings: np.ndarray, batch_size: int = 500):
        '''
        Add documents and their embeddings to the vector store, par batchs.

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
            batch_size: Nombre de documents insérés par appel à ChromaDB
        '''
        if len(documents) != len(embeddings):
            raise ValueError('Number of documents must match number of embeddings')

        total = len(documents)
        print(f'[INFO] Adding {total} documents to vector store (batch_size={batch_size})...')

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_docs = documents[start:end]
            batch_embeddings = embeddings[start:end]

            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []

            for i, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                # Generate unique ID
                doc_id = self._build_doc_id(doc, fallback_index=start + i)
                ids.append(doc_id)
                # metadatas.append(self._sanitize_metadata(dict(doc.metadata)))
                metadata = dict(doc.metadata)
                metadatas.append(metadata)
                documents_text.append(doc.page_content)
                embeddings_list.append(embedding.tolist())

            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents_text
                )
            except Exception as e:
                print(f'[ERROR] Failed to insert batch {start}-{end}: {e}')
                raise

        print(f'[INFO] Successfully added {total} documents to vector store')
        print(f'[INFO] Total documents in collection: {self.collection.count()}')

# if __name__ == '__main__':
#     csv_docs = data_loader.load_csv_documents('data/csv_data')
#     md_docs = data_loader.load_md_documents('test')
#     chunker = EmbeddingPipeline()
#     csv_chunks = chunker.chunk_documents(csv_docs, 'csv')
#     md_chunks = chunker.chunk_documents(md_docs, 'md')
#     embedded_csv = chunker.embed_chunks(csv_chunks)
#     embedded_md = chunker.embed_chunks(md_chunks)
#     chromadb_store = VectorStore()
#     chromadb_store.add_documents(csv_chunks, embedded_csv)
#     time.sleep(5)
#     print("Reprise apres 5 secondes")
#     chromadb_store.add_documents(md_chunks, embedded_md)
#     print("Documents successefuly added !!!")
