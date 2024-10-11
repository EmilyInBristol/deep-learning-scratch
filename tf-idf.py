# Install dependencies if not already installed
# !pip install transformers sklearn numpy

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Step 1: Sample knowledge base (small corpus of documents)
knowledge_base = [
    "The iPhone 14 was released in September 2022 with an improved camera and performance.",
    "The Tesla Model 3 is an electric car that offers great performance at a lower price than other Tesla models.",
    "The MacBook Air 2020 is powered by Apple's M1 chip, offering high performance with great battery life.",
    "Google's Pixel phones are known for their excellent cameras and software integration.",
    "The Samsung Galaxy S22 Ultra features a powerful camera and a high refresh rate display."
]

# Step 2: User query (a shopping question)
user_query = "Tell me about the latest iPhone and its features?"

# Step 3: Retrieval using TF-IDF to get the most relevant document from the knowledge base
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(knowledge_base)
query_vec = vectorizer.transform([user_query])

print(X.shape)
print(query_vec)

"""
# Compute cosine similarity between the query and the documents
cosine_similarities = np.dot(query_vec, X.T).toarray()[0]

# Retrieve the most relevant document based on cosine similarity
top_doc_index = np.argmax(cosine_similarities)
retrieved_doc = knowledge_base[top_doc_index]
print(f"Top retrieved document: {retrieved_doc}")

# Step 4: Load a small pre-trained GPT-2 model (you can use a smaller model for CPU efficiency)
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# Step 5: Generate a response using the LLM, augmented with the retrieved document
input_text = f"User query: {user_query}\n\nInformation retrieved: {retrieved_doc}\n\nResponse:"

# Tokenize input and generate response
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate response (you can set max_length for shorter outputs)
output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode and print the generated response
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nGenerated Response:\n{response}")
"""