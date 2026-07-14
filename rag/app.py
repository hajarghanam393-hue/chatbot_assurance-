from pathlib import Path
from src.manifest import ProcessingManifest
from src.data_loader import discover_files, load_csv_documents, load_md_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import VectorStore
from src.retrieving import RAGRetriever

if __name__ == '__main__':
    manifest = ProcessingManifest("processed_files.json")

    # --- Découverte + filtrage ---
    all_csv_files = discover_files('data/csv_data', 'csv')
    all_md_files = discover_files('data/md_data', 'md')

    new_csv_files = manifest.filter_unprocessed(all_csv_files)
    new_md_files = manifest.filter_unprocessed(all_md_files)

    print(f"[INFO] CSV: {len(new_csv_files)}/{len(all_csv_files)} fichiers à traiter")
    print(f"[INFO] MD: {len(new_md_files)}/{len(all_md_files)} fichiers à traiter")

    chunker = EmbeddingPipeline()
    chromadb_store = VectorStore()

    # --- CSV ---
    if new_csv_files:
        csv_docs = load_csv_documents(new_csv_files)
        csv_chunks = chunker.chunk_documents(csv_docs, 'csv')
        embedded_csv = chunker.embed_chunks(csv_chunks)
        chromadb_store.add_documents(csv_chunks, embedded_csv)

        # Marquer chaque fichier CSV traité (compter les chunks par fichier)
        for f in new_csv_files:
            chunk_count = sum(1 for c in csv_chunks if Path(c.metadata.get('source', '')) == f)
            manifest.mark_processed(f, chunk_count)
    else:
        print("[INFO] Aucun nouveau fichier CSV à traiter")

    # --- MD ---
    if new_md_files:
        md_docs = load_md_documents(new_md_files)
        md_chunks = chunker.chunk_documents(md_docs, 'md')
        embedded_md = chunker.embed_chunks(md_chunks)
        chromadb_store.add_documents(md_chunks, embedded_md)

        for f in new_md_files:
            chunk_count = sum(1 for c in md_chunks if Path(c.metadata.get('source', '')) == f)
            manifest.mark_processed(f, chunk_count)
    else:
        print("[INFO] Aucun nouveau fichier MD à traiter")
    
    response_generator = RAGRetriever(chromadb_store, chunker)
    query = input("Posez votre reponse s'il vous plait : ")
    while query != "exit":
        response = response_generator.response(query)
        print(response)
        query = input("Posez votre reponse s'il vous plait('exit' pour quitter le programme) : ")
