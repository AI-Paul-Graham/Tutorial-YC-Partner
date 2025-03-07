from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    """Get embedding vector for a text using Vertex AI.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector
    """
    # Instantiate the embedding model
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    
    # Prepare input for the embedding
    embedding_input = TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT")
    
    # Fetch embedding (this returns a list; we take [0] since we have 1 input)
    embedding_obj = model.get_embeddings([embedding_input])[0]
    
    # Extract the actual values from the TextEmbedding object
    embedding_vector = embedding_obj.values
    
    return embedding_vector

if __name__ == "__main__":
    # Test the embedding function
    sample_text = "This is a test sentence to embed."
    embedding = get_embedding(sample_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First few values: {embedding[:5]}") 