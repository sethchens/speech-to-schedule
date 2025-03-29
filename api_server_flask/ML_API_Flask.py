from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import notebook_login
import torch
import numpy as np

# Initialize Flask app and Flask-RESTful API
app = Flask(__name__)
api = Api(app)

# Load the pre-trained DistilBERT model and tokenizer once
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Similarity calculation function
def compute_similarity(resume_text, job_description_text):
    # Tokenize the input texts
    tokenized_resume = tokenizer(resume_text, return_tensors='pt', padding=True, truncation=True)
    tokenized_job = tokenizer(job_description_text, return_tensors='pt', padding=True, truncation=True)
    
    print(f"Tokenized Resume: {tokenized_resume}")
    print(f"Tokenized Job: {tokenized_job}")

    # Get the embeddings for both inputs (without gradient descent)
    with torch.no_grad():
        tokenized_resume = model(**tokenized_resume)
        tokenized_job = model(**tokenized_job)

    print(f"Tokenized Resume Embeddings: {tokenized_resume.last_hidden_state}")
    print(f"Tokenized Job Embeddings: {tokenized_job.last_hidden_state}")

    # Get the embeddings
    embedding_resume = tokenized_resume.last_hidden_state.mean(dim=1)
    embedding_job = tokenized_job.last_hidden_state.mean(dim=1)

    print(f"Embedding Resume: {embedding_resume}")
    print(f"Embedding Job: {embedding_job}")

    cos_sim = torch.nn.functional.cosine_similarity(embedding_resume, embedding_job)
    print(f"Cosine Similarity: {cos_sim.item()}")
    
    return cos_sim.item()  # Return the similarity score

# API endpoint to compute similarity score
class Similarity_score_endpoint(Resource):
    def post(self):
        if request.is_json:
            data = request.get_json()  # Get input data from JSON (resume and job description)
        else:
            return jsonify({'error': 'Request must be JSON'}), 400
        # Ensure the required keys are present in the input data
        if 'resume' not in data or 'job_description' not in data:
            return jsonify({'error': 'Missing resume or job_description'}), 400

        resume = data['resume']
        job_description = data['job_description']
        
        similarity_score = compute_similarity(resume, job_description)
        return jsonify({'similarity_score': similarity_score})

class Speech_to_schedule_endpoint(Resource):
    model_test = AutoModelForSeq2SeqLM.from_pretrained('sethchens/t5-speech-to-schedule')
    tokenizer = AutoTokenizer.from_pretrained('sethchens/t5-speech-to-schedule')

    def post(self):
        if request.is_json:
            data = request.get_json()  # Get input data from JSON (resume and job description)
        else:
            return jsonify({'error': 'Request must be JSON'}), 400


api.add_resource(Similarity_score_endpoint, '/similarity_score')

if __name__ == '__main__':
    model_test = AutoModelForSeq2SeqLM.from_pretrained('sethchens/t5-speech-to-schedule')
    tokenizer_test = AutoTokenizer.from_pretrained('sethchens/t5-speech-to-schedule')
    input_text = "I have a meeting at church today at 7 pm"
    inputs = tokenizer_test(input_text, return_tensors="pt")
    outputs = model_test.generate(inputs.input_ids, max_new_tokens=100) #added max_new_tokens.
    decoded_text = tokenizer_test.decode(outputs[0], skip_special_tokens=True)
    print(decoded_text)

    # app.run(debug=True, host='0.0.0.0')

