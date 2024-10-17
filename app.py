from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained question-answering model from Hugging Face
qa_model = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad")

# Define the /qa endpoint
@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get('question')
    context = data.get('context')

    if not question or not context:
        return jsonify({"error": "Both 'question' and 'context' fields are required."}), 400

    # Run the QA model
    result = qa_model(question=question, context=context)

    return jsonify({
        "question": question,
        "context": context,
        "answer": result['answer']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
