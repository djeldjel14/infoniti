from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import time
import PyPDF2

# Télécharger les stopwords
nltk.download('stopwords')

# Fonction pour extraire le texte d'un fichier PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

app = Flask(__name__)

# Charger les documents dans un dossier "documents"
DOCUMENTS_FOLDER = "documents"
documents = {}

for filename in os.listdir(DOCUMENTS_FOLDER):
    file_path = os.path.join(DOCUMENTS_FOLDER, filename)
    
    # Lire les fichiers texte
    if filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents[filename] = f.read()
    
    # Lire les fichiers PDF
    elif filename.endswith(".pdf"):
        documents[filename] = extract_text_from_pdf(file_path)

# Prétraitement des textes
stop_words = set(stopwords.words('french'))
def preprocess(text):
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Appliquer le prétraitement
processed_docs = {filename: preprocess(content) for filename, content in documents.items()}

# Vectorisation avec TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs.values())
filenames = list(processed_docs.keys())

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start_time = time.time()
        query = request.form["query"]
        processed_query = preprocess(query)
        query_vector = vectorizer.transform([processed_query])

        # Calculer la similarité cosinus
        scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()

        # Trier les résultats
        ranked_results = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

        # Générer les extraits pertinents
        results = [(filename, score, extract_snippet(documents[filename], query)) for filename, score in ranked_results if score > 0]

        execution_time = time.time() - start_time

        return render_template("resultats.html", results=results, execution_time=execution_time)

    # Si la requête est GET, afficher la page de recherche
    return render_template("index.html")

    results = []
    execution_time = 0
    if request.method == "POST":
        start_time = time.time()
        query = request.form["query"]
        processed_query = preprocess(query)
        query_vector = vectorizer.transform([processed_query])

        # Calculer la similarité cosinus
        scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()

        # Trier les résultats
        ranked_results = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

        # Générer les extraits pertinents
        for filename, score in ranked_results:
            if score > 0:
                snippet = extract_snippet(documents[filename], query)
                results.append((filename, score, snippet))

        execution_time = time.time() - start_time

    return render_template("resultats.html", results=results, execution_time=execution_time)

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    """ Retourne une liste de suggestions basées sur l'input utilisateur """
    query = request.args.get("q", "").lower()
    suggestions = [word for word in vectorizer.get_feature_names_out() if query in word][:10]
    return jsonify(suggestions)

def extract_snippet(text, query):
    """ Extrait une phrase contenant le mot-clé """
    sentences = text.split('.')
    query_words = query.lower().split()
    
    for sentence in sentences:
        if any(word in sentence.lower() for word in query_words):
            return f"...{sentence.strip()}..."
    
    return text[:200] + "..."  

# 📌 🔥 **Ajout des routes pour afficher et télécharger les documents**
@app.route('/document/<nom_fichier>')
def afficher_document(nom_fichier):
    chemin_fichier = os.path.join(DOCUMENTS_FOLDER, nom_fichier)
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            contenu = f.read()
        return f"<h1>{nom_fichier}</h1><pre>{contenu}</pre><br><a href='/download/{nom_fichier}'>Télécharger</a>"
    except FileNotFoundError:
        return "<h1>Document non trouvé</h1>", 404

@app.route('/download/<nom_fichier>')
def telecharger_document(nom_fichier):
    return send_from_directory(DOCUMENTS_FOLDER, nom_fichier, as_attachment=True)

# 🔥 **Exécuter l'application**
if __name__ == "__main__":
    app.run(debug=True)
