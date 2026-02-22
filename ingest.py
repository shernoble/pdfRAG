import fitz
import tiktoken 
from openai import OpenAI
import faiss
import pickle
import numpy as np
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
# load the documents

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# chunk the text
def chunk_text(txt, chunk_size = 500, overlap = 100):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(txt)

    chunks = []
    for i in range(0,len(tokens), chunk_size - overlap):

        chunk = tokens[i : i + chunk_size]
        chunks.append(encoding.decode(chunk))
    
    return chunks
        

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# generate embeddings
def generate_embeddings(txt):
    return embedding_model.encode(txt)

# store the embeddings
def store_embeddings_index(embeddings, chunks):

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    vectors = np.array(embeddings).astype("float32")
    index.add(vectors) #saves to index

    faiss.write_index(index, "vector.index") #saves to disk

    # need to store the chunks as well
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    # run doc ingestion pipeline
    pdf_text = extract_text_from_pdf("doc1.pdf")
    chunked_text = chunk_text(pdf_text)
    embeddings = generate_embeddings(chunked_text)

    store_embeddings_index(embeddings, chunked_text)

    print("Index built successfully")

