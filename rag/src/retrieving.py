from src.embedding import EmbeddingPipeline
from src.vector_store import VectorStore
from typing import Any
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

class RAGRetriever:
    '''Handles query based retrieavl from the vector store'''

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingPipeline):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(groq_api_key=self.groq_api_key, model='llama-3.3-70b-versatile', temperature=0.1, max_tokens=1024)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0, categorie: str = None) -> list[dict[str, Any]]:
        print(f'Retrieving documents for query: "{query}"')
        print(f'Top K : {top_k}, Score threshold: {score_threshold}, Categorie: {categorie}')

        query_embedding = self.embedding_manager.embed_query(query)

        try:
            query_kwargs = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k
            }
            if categorie:
                query_kwargs["where"] = {"categorie": categorie}

            results = self.vector_store.collection.query(**query_kwargs)

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                print(f'Retrieved {len(retrieved_docs)} documents (after filtering)')
            else:
                print('No documents found')
            
            return retrieved_docs
    
        except Exception as e:
            print(f'Error during retrieval: {e}')
            return []

    def response(self, query, categorie: str = "Sinistres"):
        # 1. Recherche large, restreinte a la categorie Sinistres, pour identifier le document le plus pertinent
        initial_results = self.retrieve(query, top_k=5, categorie=categorie)

        if not initial_results:
            return "No relevant context found to answer the question."

        best_doc_id = initial_results[0]['metadata'].get('document_id')

        if best_doc_id:
            # 2. Recuperer TOUTES les sections de ce document
            full_doc = self.vector_store.collection.get(where={"document_id": best_doc_id})
            chunks = list(zip(full_doc['documents'], full_doc['metadatas']))
            chunks.sort(key=lambda x: x[1].get('chunk_index', 0))
            results = [{'content': c, 'metadata': m} for c, m in chunks]
        else:
            results = initial_results

        print(f"\n=== {len(results)} sections du document {best_doc_id} ===")
        for doc in results:
            titre = doc['metadata'].get('titre', '?')
            print(f"\n[chunk_index={doc['metadata'].get('chunk_index')}] {titre}")
            print(doc['content'][:300])
        print("=== FIN DEBUG ===\n")

        context = "\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return "No relevant context found to answer the question."
        
        prompt = f'''Use the following context to answer the question conciesly.
            Context:
            {context}

            Question:
            {query}

            Answer:'''
        
        response = self.llm.invoke(prompt)
        return response.content
