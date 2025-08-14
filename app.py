import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURACIÓN ---
# Render usará "variables de entorno" para las llaves secretas.
# Es más seguro que pegarlas directamente en el código.
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Configuración del cerebro (Gemini)
genai.configure(api_key=GOOGLE_API_KEY)
model_gen = genai.GenerativeModel('gemini-1.5-flash-latest')
print("Conexión con Gemini establecida.")

# --- CARGAR LA MEMORIA PERSISTENTE ---
print("\nCargando memoria persistente desde archivos...")
index = faiss.read_index("ai_memory.index")
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base_nested = json.load(f)
print("¡Archivos de conocimiento cargados exitosamente!")

def smart_flatten(data, parent_source=''):
    chunks = []
    if isinstance(data, dict):
        if 'descripcion' in data:
            source_parts = [parent_source];
            if 'etiqueta' in data: source_parts.append(f"Etiqueta {data['etiqueta']}")
            source = ' -> '.join(filter(None, source_parts)); chunks.append({'source': source, 'content': data['descripcion']})
        for key, value in data.items():
            if key != 'descripcion':
                new_source = f"{parent_source} -> {key}" if parent_source else key; chunks.extend(smart_flatten(value, new_source))
    elif isinstance(data, list):
        for item in data: chunks.extend(smart_flatten(item, parent_source))
    elif isinstance(data, str) and parent_source: chunks.append({'source': parent_source, 'content': data})
    return chunks

knowledge_chunks = smart_flatten(knowledge_base_nested)
print(f"Memoria procesada con {len(knowledge_chunks)} fichas.")


# --- APLICACIÓN FLASK ---
app = Flask(__name__)
CORS(app)

def get_ai_response(question):
    question_embedding = model_emb.encode([question], convert_to_numpy=True)
    k = 5
    distances, indices = index.search(question_embedding, k)
    relevant_chunks = [knowledge_chunks[i]['content'] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
    prompt = f"Eres un mentor experto en desarrollo web. Basándote ÚNICAMENTE en el siguiente contexto, responde. Contexto: --- {context} --- Pregunta: {question}. Tu respuesta:"
    response = model_gen.generate_content(prompt)
    return response.text

@app.route("/")
def home():
    return "<h1>El servidor Backend de la IA está vivo.</h1>"

@app.route("/ask")
def ask_api():
    user_question = request.args.get('query')
    if not user_question:
        return jsonify({"error": "Falta el parámetro 'query'."}), 400

    try:
        ai_answer = get_ai_response(user_question)
        return jsonify({"pregunta": user_question, "respuesta": ai_answer})
    except Exception as e:
        print(f"!!! ERROR INTERNO: {e}")
        return jsonify({"error": "Ocurrió un error interno en el servidor.", "detalle": str(e)}), 500
