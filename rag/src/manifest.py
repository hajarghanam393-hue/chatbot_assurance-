import json
import hashlib
from pathlib import Path
from datetime import datetime


class ProcessingManifest:
    '''Garde une trace des fichiers déjà traités pour éviter de refaire load/chunk/embed/store.'''

    def __init__(self, manifest_path: str = "processed_files.json"):
        self.manifest_path = Path(manifest_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _file_hash(file_path: Path) -> str:
        '''Hash du contenu (plus fiable que mtime, insensible aux copies/sync cloud).'''
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def is_processed(self, file_path: Path) -> bool:
        '''True si le fichier a déjà été traité ET n'a pas changé depuis.'''
        key = str(file_path.resolve())
        if key not in self.data:
            return False
        return self.data[key]['hash'] == self._file_hash(file_path)

    def mark_processed(self, file_path: Path, chunk_count: int):
        key = str(file_path.resolve())
        self.data[key] = {
            'hash': self._file_hash(file_path),
            'chunk_count': chunk_count,
            'processed_at': datetime.now().isoformat()
        }
        self._save()

    def filter_unprocessed(self, files: list[Path]) -> list[Path]:
        '''Retourne uniquement les fichiers nouveaux ou modifiés depuis le dernier passage.'''
        return [f for f in files if not self.is_processed(f)]