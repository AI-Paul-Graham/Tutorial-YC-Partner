import numpy as np
import faiss
import os

def create_index(vectors, save_path=None):
    """Create a FAISS index from vectors.
    
    Args:
        vectors (numpy.ndarray): Array of vectors to index (float32 type)
        save_path (str, optional): Path to save the index
        
    Returns:
        faiss.Index: The created FAISS index
    """
    # Convert to numpy array if not already
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors, dtype=np.float32)
    else:
        # Ensure vectors are float32 type as required by FAISS
        vectors = vectors.astype(np.float32)
    
    # Make sure vectors is a 2D array with shape (n, dim)
    if len(vectors.shape) == 1:
        vectors = vectors.reshape(1, -1)
    
    print(f"Vectors shape for index creation: {vectors.shape}")
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    
    # Get dimensions
    d = vectors.shape[1]
    
    # Create index that uses inner product (for cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(d)
    
    # Add vectors to index
    index.add(vectors)
    print(f"Added {index.ntotal} vectors to index")
    
    # Save index if path provided
    if save_path:
        save_index(index, save_path)
    
    return index

def search_index(query_vector, index, top_k=5):
    """Search the FAISS index for similar vectors.
    
    Args:
        query_vector (list/array): The query embedding vector
        index (faiss.Index): FAISS index to search
        top_k (int): Number of top results to return
        
    Returns:
        tuple: (scores, indices) where scores contains similarity scores 
               and indices contains the corresponding indices in the original vector list
    """
    # Convert to numpy array if not already
    if not isinstance(query_vector, np.ndarray):
        query_vector = np.array(query_vector, dtype=np.float32)
    else:
        # Ensure vector is float32 type as required by FAISS
        query_vector = query_vector.astype(np.float32)
    
    # Make sure query_vector is a 2D array with shape (1, dim)
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Normalize query vector for cosine similarity
    faiss.normalize_L2(query_vector)
    
    # Search the index
    scores, indices = index.search(query_vector, top_k)
    
    # Return raw scores and indices
    return scores[0], indices[0]

def save_index(index, path):
    """Save a FAISS index to disk.
    
    Args:
        index (faiss.Index): FAISS index to save
        path (str): Path to save the index
    """
    faiss.write_index(index, path)

def load_index(path):
    """Load a FAISS index from disk.
    
    Args:
        path (str): Path to the saved index
        
    Returns:
        faiss.Index: The loaded FAISS index
    """
    return faiss.read_index(path)

if __name__ == "__main__":
    # Test the vector search functions with FAISS
    import random
    
    # Create sample embeddings
    dim = 1536  # Dimension for OpenAI embeddings
    num_vectors = 100
    sample_embeddings = np.array([
        [random.random() for _ in range(dim)] for _ in range(num_vectors)
    ]).astype(np.float32)
    
    print(f"Sample embeddings shape: {sample_embeddings.shape}")
    
    # Create an index
    index = create_index(sample_embeddings)
    
    # Create a query - using a 1D array to test shape handling
    query = np.array([random.random() for _ in range(dim)], dtype=np.float32)
    print(f"Query shape before search: {query.shape}")
    
    # Search
    scores, indices = search_index(query, index, top_k=5)
    
    print(f"Results:")
    for i, idx in enumerate(indices):
        if idx >= 0:  # Valid index
            print(f"  Result {i+1}: Score: {scores[i]:.4f}, Index: {idx}")
    
    # Test save and load (optional)
    test_dir = "test_faiss"
    os.makedirs(test_dir, exist_ok=True)
    
    index_path = os.path.join(test_dir, "test_index.faiss")
    
    save_index(index, index_path)
    loaded_index = load_index(index_path)
    
    print("\nTesting loaded index:")
    loaded_scores, loaded_indices = search_index(query, loaded_index, top_k=5)
    for i, idx in enumerate(loaded_indices):
        if idx >= 0:  # Valid index
            print(f"  Result {i+1}: Score: {loaded_scores[i]:.4f}, Index: {idx}")
            
    # Example: Using search results to get original vectors
    print("\nRetrieving original vectors by index:")
    for i, idx in enumerate(indices[:3]):  # Just show first 3
        if idx >= 0:
            original_vector = sample_embeddings[idx]
            # Just print the first 5 elements of the vector for brevity
            print(f"  Result {i+1}: Index {idx}, Vector preview: {original_vector[:5]}") 