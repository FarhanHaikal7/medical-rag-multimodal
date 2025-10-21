from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import os
import logging
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss
import Levenshtein
import bert_score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#--------------------
# Load all models and data once at startup
#--------------------

print("Loading models and data...")

# --- Main Models ---
model_name = "FarhanHaikal/gpt2-lora-small"  # Path to your fine-tuned GPT2 model
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
print("GPT2 model loaded.")

# --- Retrieval Models & Data ---
df = load_dataset('json', data_files='medquad_cleaned_small.jsonl', split='train')
corpus = df["answer"]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
print("MedQuAD dataset and TF-IDF Vectorizer loaded and fitted.")

# Load the FAISS index for text
# This is now loaded once and used by the retrieve function globally.
index_text = faiss.read_index("medquad_embedder_small.index")
print("FAISS index for text loaded.")

# Load the embedder for text retrieval
embedder = SentenceTransformer('all-mpnet-base-v2')
print("SentenceTransformer retrieval embedder loaded.")

# --- Evaluation Model ---
# OPTIMIZATION: Load the evaluation embedder only once at startup.
eval_embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer evaluation embedder loaded.")

print("All models and data loaded successfully.")
#--------------------
# Define utility functions (for RAG)
#--------------------

def retrieve(query, top_k=3):
    """Retrieves top_k answers from the FAISS index."""
    answers = df["answer"]    
    query_embed = embedder.encode(query)
    query_vec = np.array([query_embed]).astype(np.float32)
    
    # OPTIMIZATION: Use the globally loaded 'index_text' instead of reloading from disk.
    _, I = index_text.search(query_vec, top_k)
    
    return [answers[i] for i in I[0]]

def choose_expected_response(prompt, dataset):
    """Finds the most similar question in the dataset to use its answer as a reference."""
    all_texts = [prompt] + list(dataset['question'])
    tfidf_matrix = vectorizer.transform(all_texts) # Use transform instead of fit_transform
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_index = similarities.argmax()
    return dataset['answer'][best_index]

def get_embedding(text, embedder_model):
    """Gets a sentence embedding for comparison using the provided model."""
    # OPTIMIZATION: Use the passed-in model instead of reloading it.
    return embedder_model.encode(text.strip().lower())

def comparison_evaluator(response, expected):
    """Compares the generated response to the expected answer."""
    response, expected = response.strip().lower(), expected.strip().lower()
    
    # Levenshtein Similarity
    lev_distance = Levenshtein.distance(response, expected)
    if max(len(response), len(expected)) == 0:
        lev_similarity = 1.0
    else:
        lev_similarity = max(0, 1 - lev_distance / max(len(response), len(expected)))
    
    # Cosine Similarity
    # OPTIMIZATION: Use the globally loaded evaluation embedder.
    emb1 = get_embedding(response, eval_embedder)
    emb2 = get_embedding(expected, eval_embedder)
    cos_sim = cosine_similarity([emb1], [emb2])[0][0]
    
    # BERTScore
    P, R, F1 = bert_score.score([response], [expected], lang="en")
    
    return lev_similarity, cos_sim, P.mean().item(), R.mean().item(), F1.mean().item()

def rag(query, max_context_length=300):
    """Performs the full Retrieval-Augmented Generation process."""
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)[:max_context_length]
    
    # Use the main 'df' dataset to find the reference answer
    reference_answer = choose_expected_response(query, df)
    
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = gpt2_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs['input_ids'], 
            max_new_tokens=200, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2
        )
        
    generated_answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_answer.split("Answer:")[-1].strip()
    
    return answer, reference_answer

#--------------------
# Define Flask routes
#--------------------

@app.route('/')
def home():
    """Serves the main HTML page."""
    # I've changed this to 'index.html' to match the simplified HTML I provided.
    # If your file is named 'indexquery.html', you can change it back.
    return render_template('indexquery.html') 

@app.route('/query', methods=['POST'])
def query():
    """API endpoint for text queries."""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing "query" in request body'}), 400
        
    query_text = data.get('query')
    
    try:
        # Call the RAG function
        generated_answer, reference_answer = rag(query_text)
        
        # Evaluate the answer
        lev_similarity, cos_sim, precision, recall, f1_score = comparison_evaluator(generated_answer, reference_answer)
        
        # Return the full JSON response
        return jsonify({
            'generated_answer': generated_answer,
            'reference_answer': reference_answer,
            'lev_similarity': float(lev_similarity),
            'cosine_similarity': float(cos_sim),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'question': query_text
        })
    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}")
        return jsonify({'error': f'Failed to process query: {str(e)}'}), 500

if __name__ == '__main__':
  app.run(debug=True)
