I. RAG Setup:
1. Se deplacer vers le dossier rag
    cd chemin/vers/dossier/rag
2. Créer un environnement virtuel isolé
    python -m venv venv
3. Installer les bibliotheques Python
    pip install -r requirements.txt
4. Lancer le pipline RAG(chunking, embedding, ...)
    python app.py
5. Lancer le serveur web local (Backend)
    uvicorn api:app --reload --port 8000
II. Angular Setup:
1. Installer node.js
2. Installer Angular CLI
    npm install -g @angular/cli
3. Verifier l'installation d'Angular
    ng version
4. Lancer le serveur web local (Frontend)
    ng serve