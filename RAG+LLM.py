from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# Sample documents for retrieval
documents = [
    "The capital of France is Paris. It is known for its art, culture, and history.",
    "The Eiffel Tower is one of the most famous landmarks in the world, located in Paris, France.",
    "The Louvre is a world-renowned art museum located in Paris, housing works such as the Mona Lisa.",
    "France is known for its cuisine, especially bread, cheese, and wine.",
    "Paris is known as the City of Lights due to its role in the Age of Enlightenment."
]

# Initialize the retriever (Sentence-BERT for semantic similarity search)
retriever = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all documents into vectors
document_embeddings = retriever.encode(documents, convert_to_tensor=True)

# Initialize the generator (Hugging Face language model)
generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Function to perform retrieval
def retrieve_documents(query, k=2):
    # Encode the query
    query_embedding = retriever.encode(query, convert_to_tensor=True)

    # Compute similarity between query and documents
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)

    # Get the top k documents based on similarity
    top_k_indices = torch.topk(similarities, k=k).indices

    # Return the top k documents
    return [documents[i] for i in top_k_indices[0]]

# Function to generate a response based on retrieved documents
def generate_response(query):
    # Retrieve the top 2 relevant documents
    retrieved_docs = retrieve_documents(query, k=2)

    # Combine the retrieved documents into a context for generation
    context = " ".join(retrieved_docs)

    # Create a prompt by combining query and context
    input_text = f"Question: {query}\nContext: {context}"

    # Tokenize the input
    inputs = generator_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate a response
    summary_ids = generator_model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)

    # Decode and return the generated response
    return generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example query
query = "What is Paris known for?"

# Generate and print the response using the RAG system
response = generate_response(query)
print(f"Query: {query}")
print(f"Response: {response}")
