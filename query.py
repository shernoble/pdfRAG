# query pipeline
# user-> query-> query_embedding -> L2 similarity search with index -> returns relevant chunks -> passed to llm -> reasons and returns answer

import numpy as np
import faiss
import pickle
import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# load index
client = Groq(
    api_key = os.environ.get("GROQ_API_KEY")
)

index = faiss.read_index("vector.index")

with open("chunks.pkl","rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# get user query
def embed_query(query):
    return embedding_model.encode(query)

def retrieve(query, k = 3):
    # similarity search with index
    query_vector = embed_query(query)

    query_vector = np.expand_dims(query_vector, axis=0)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k)

    return [chunks[i] for i in indices[0]]

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a helpful assistant.
    Answer ONLY from the provided context.
    If the answer is not in context, say "Not found in document."

    Context:
    {context}

    Question:
    {question}
    """
    return prompt

def generate_answer(prompt):

    reponse = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [
            { 
                "role" : "user" , 
                "content" : prompt 
            }
        ]
    )
    # print(reponse)
    return reponse.choices[0].message.content

if __name__ == "__main__":
    query = input("what is your question?")

    relevant_chunks = retrieve(query)
    prompt = build_prompt(relevant_chunks, query)

    print(generate_answer(prompt))