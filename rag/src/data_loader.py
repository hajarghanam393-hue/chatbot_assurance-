from pathlib import Path
from typing import Any
from langchain_community.document_loaders import CSVLoader
import re
from tqdm import tqdm
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
import csv

def discover_files(data_dir: str, extension: str) -> list[Path]:
    """Liste tous les fichiers d'une extension donnée dans data_dir."""
    data_path = Path(data_dir).resolve()
    return list(data_path.glob(f'**/*.{extension}'))

def load_csv_documents(csv_files: list[Path]) -> list[Any]:
    """
    Load all csv files from the data directory and convert to LangChain document structure.
    """
    documents = []
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file), encoding='utf-8')
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")

            # Extraction des valeurs de la 1ère colonne (peu importe son nom réel)
            # pour les utiliser comme record_id, normalisé quel que soit le fichier CSV
            with open(csv_file, encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                first_col_name = header[0]
                first_col_values = [row[0] for row in reader if row]

            if len(first_col_values) != len(loaded):
                print(f"[WARN] Mismatch entre le nombre de lignes CSV ({len(first_col_values)}) "
                      f"et les documents chargés ({len(loaded)}) pour {csv_file} — record_id non assigné")
            else:
                for doc, record_id in zip(loaded, first_col_values):
                    doc.metadata["record_id"] = record_id
                    doc.metadata["record_id_column"] = first_col_name  # utile pour debug/traçabilité

            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")
    return documents


def _parse_frontmatter(raw_yaml: str) -> dict:
    """
    Parse un frontmatter simple 'Clé: valeur' ligne par ligne,
    plus robuste que yaml.safe_load face aux ':' non échappés
    dans les valeurs (ex: titres contenant ':').
    """
    metadata = {}
    for line in raw_yaml.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, _, value = line.partition(':')
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata

def load_md_documents(md_files: list[Path]) -> list[Any]:
    """
    Load all markdown files from the data directory, parse le frontmatter
    et le contenu, et convertit en structure Document LangChain.
    """
    documents = []

    print(f"[DEBUG] Found {len(md_files)} MD files")

    frontmatter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)', re.DOTALL)

    for md_file in tqdm(md_files, desc="Loading MD files"):
        try:
            try:
                text = md_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                text = md_file.read_text(encoding='utf-8-sig', errors='replace')

            match = frontmatter_pattern.match(text)

            if match:
                raw_yaml, content = match.groups()
                metadata = _parse_frontmatter(raw_yaml)
            else:
                metadata = {}
                content = text

            clean_metadata = {
                "source": str(md_file),
                "document_id": metadata.get("DocumentID", ""),
                "titre": metadata.get("Titre", ""),
                "categorie": metadata.get("Catégorie", ""),
                "tags": metadata.get("Tags", ""),
                "date": str(metadata.get("Date", "")),
                "auteur": metadata.get("Auteur", ""),
                "version": str(metadata.get("Version", "")),
                "mots_cles": metadata.get("Mots-clés", ""),
            }

            documents.append(Document(page_content=content.strip(), metadata=clean_metadata))

        except Exception as e:
            print(f"[ERROR] Failed to load MD {md_file}: {e}")

    print(f"[DEBUG] Loaded {len(documents)} MD documents")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_md_documents('test')
    prompt = 'Bienvenue dans le test!'
    while prompt != 'n':
        index = int(input("Saisir l'indice :"))
        print(docs[index])
        prompt = input("Refaire le test ? y/n")
        
