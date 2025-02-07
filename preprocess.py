import os
import pandas as pd
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

def chunk_indices(n, chunk_size=10000):
    """
    Given the size of a text (n characters), split into chunks of length `chunk_size`.
    If the final chunk is < 50% of chunk_size, merge it with the previous chunk.
    Returns a list of (start_index, end_index) pairs.
    """
    if n <= 0:
        return []

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((start, end))
        start = end

    # Merge the last chunk if it's too small
    if len(chunks) > 1:
        last_start, last_end = chunks[-1]
        last_chunk_size = last_end - last_start
        if last_chunk_size < (0.5 * chunk_size):
            second_last_start, second_last_end = chunks[-2]
            merged_chunk = (second_last_start, last_end)
            chunks[-2] = merged_chunk
            chunks.pop()

    return chunks

# 1. Instantiate the embedding model
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# 2. Prepare a list to hold the results
records = []

# 3. Loop through each text_id file
for text_id in range(1, 355):
    file_path = f"./data/{text_id}.txt"
    
    # Skip if file doesn't exist
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}, skipping.")
        continue
    
    # Read file content
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    # 4. Compute the chunk boundaries
    intervals = chunk_indices(len(text_content), chunk_size=10000)

    # 5. For each chunk, get its embedding
    for chunk_id, (start, end) in enumerate(intervals, start=1):
        chunk_text = text_content[start:end]

        # Prepare input for the embedding
        embedding_input = TextEmbeddingInput(chunk_text, "RETRIEVAL_DOCUMENT")
        
        # Fetch embedding (this returns a list; we take [0] since we have 1 input)
        embedding_vector = model.get_embeddings([embedding_input])[0]

        # 6. Append to our records
        records.append({
            "text_id": text_id,
            "chunk_id": chunk_id,
            "embedding": embedding_vector  # embedding_vector is a list of floats
        })

# 7. Convert list of dicts to DataFrame
df = pd.DataFrame(records, columns=["text_id", "chunk_id", "embedding"])

# 8. Save to CSV
df.to_csv("embeddings.csv", index=False)

print(df.head())
