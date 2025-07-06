from flask import Flask, request, jsonify
import json
import boto3
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sagemaker.huggingface import HuggingFacePredictor

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        query = request.json.get("query", "")
        if not query:
            return jsonify({"error": "Missing query"}), 400

        # Load embedding model
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_vec = embed_model.encode(query)

        # Download vectors from S3
        s3 = boto3.client("s3")
        bucket = "rag-b01013736"
        key = "vectors/knowledge_vectors.json"
        local_path = "/tmp/knowledge_vectors.json"
        s3.download_file(bucket, key, local_path)

        with open(local_path, "r") as f:
            data = json.load(f)
        kb_vectors = np.array(data["vectors"])
        kb_chunks = data["text_chunks"]

        # Vector Search
        scores = [(1 - cosine(query_vec, v), i) for i, v in enumerate(kb_vectors)]
        top_chunks = sorted(scores, reverse=True)[:3]
        context = "\n\n".join([kb_chunks[i] for _, i in top_chunks])

        # Prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate Answer
        predictor = HuggingFacePredictor(endpoint_name="flan-t5-small-endpoint")
        response = predictor.predict({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 150, "temperature": 0.7}
        })

        answer = response[0]["generated_text"]
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
