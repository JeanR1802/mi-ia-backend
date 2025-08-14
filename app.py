import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURACIÓN ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model_gen = genai.GenerativeModel('gemini-1.5-flash-latest')
print("Conexión con Gemini establecida.")

# --- CARGAR LA MEMORIA PERSISTENTE ---
print("\nCargando memoria persistente...")
index = faiss.read_index("ai_memory.index")
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base_nested = json.load(f)

# (Función de aplanamiento sin cambios)
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

@app.route("/")
def home():
    return "<h1>Servidor Backend de la IA (ligero) está vivo.</h1>"

@app.route("/ask-vector", methods=['POST'])
def ask_api_vector():
    data = request.get_json()
    if not data or 'vector' not in data:
        return jsonify({"error": "Falta el vector en la petición."}), 400
    
    try:
        # Recibimos el vector del frontend y lo usamos para buscar
        user_vector = np.array(data['vector']).astype('float32').reshape(1, -1)
        k = 5
        distances, indices = index.search(user_vector, k)
        
        relevant_chunks = [knowledge_chunks[i]['content'] for i in indices[0]]
        context = "\n\n".join(relevant_chunks)
        
        # El prompt es un poco diferente porque ya no sabemos la pregunta original
        prompt = f"Contexto: --- {context} --- Pregunta Original (no la sabes, solo el contexto): Basado en el contexto, genera una respuesta útil y directa."
        
        response = model_gen.generate_content(prompt)
        return jsonify({"respuesta": response.text})
    except Exception as e:
        print(f"!!! ERROR INTERNO: {e}")
        return jsonify({"error": "Ocurrió un error interno.", "detalle": str(e)}), 500
