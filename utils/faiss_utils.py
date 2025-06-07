import faiss
import pandas as pd
import numpy as np

#query structures database- DTC INFO
def query_faiss_index(query_text, index, model, top_k=1):
    query_embedding = model.encode(query_text)  # Get embedding of the query
    query_embedding = np.array([query_embedding])  # Convert to numpy array

    # Perform the search in the FAISS index
    distances, indices = index.search(query_embedding, top_k)  # Get the top_k most similar entries

    return distances, indices

#query from manuals
def query_model_manuals(query_text, model, index, texts, top_k=20):
    query_embedding = model.encode([query_text], convert_to_tensor=True).cpu().detach().numpy()
    
    # Perform the search in the FAISS index
    distances, indices = index.search(query_embedding, top_k)  # Get the top_k most similar entries
    
    matching_texts = [texts[i] for i in indices[0]]
    return matching_texts