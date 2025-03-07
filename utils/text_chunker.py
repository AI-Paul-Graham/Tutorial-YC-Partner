def chunk_text(text, chunk_size=2000, overlap=500):
    """Split text into overlapping chunks of approximately chunk_size chars.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Target size of each chunk in characters
        overlap (int): Number of overlapping characters between chunks
    
    Returns:
        list: List of text chunks
    """
    # Guard against invalid parameters that would cause infinite loops
    if overlap >= chunk_size:
        overlap = chunk_size // 2  # Default to 50% overlap if invalid
        
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text) and end - start == chunk_size:
            # Try to find a good breaking point
            last_period = text.rfind('.', start, end)
            if last_period > start + 0.5 * chunk_size:
                end = last_period + 1
        
        chunks.append(text[start:end])
        
        # Ensure we make forward progress
        if end == len(text):
            # We've reached the end of the text
            break
            
        # Move forward by at least 1 character if overlap would prevent progress
        start = min(end, start + max(1, end - start - overlap))

    return chunks

if __name__ == "__main__":
    # Test the chunking function
    sample_text = "This is a test paragraph. It contains several sentences. " * 10
    chunks = chunk_text(sample_text, chunk_size=2000, overlap=500)
    
    print(f"Original text length: {len(sample_text)}")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (length {len(chunk)}): {chunk[:50]}...") 