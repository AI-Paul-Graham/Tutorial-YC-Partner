from pocketflow import Node, BatchNode, Flow
import os
import json
import numpy as np
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("paul_graham_rag")

from utils.call_llm import call_llm
from utils.data_loader import load_data
from utils.text_chunker import chunk_text
from utils.embedding import get_embedding
from utils.vector_search import create_index, save_index, load_index, search_index
from utils.text_to_speech import synthesize_text_to_speech

# -------------------- Offline Processing Nodes --------------------

class LoadEssaysNode(Node):
    def prep(self, shared):
        logger.info("LoadEssaysNode: Preparing to load essays and metadata")
        return shared["data_dir"], shared["meta_csv"]
    
    def exec(self, inputs):
        data_dir, meta_csv = inputs
        logger.info(f"LoadEssaysNode: Loading essays and metadata from {data_dir} and {meta_csv}")
        essays, metadata = load_data(data_dir, meta_csv)
        logger.info(f"LoadEssaysNode: Loaded {len(essays)} essays")
        return essays, metadata
    
    def post(self, shared, prep_res, exec_res):
        essays, metadata = exec_res
        shared["essays"] = essays
        shared["metadata"] = metadata
        logger.info(f"LoadEssaysNode: Stored {len(essays)} essays and metadata in shared store")
        return "default"


class ChunkTextNode(BatchNode):
    def prep(self, shared):
        essays = shared["essays"]
        metadata = shared["metadata"]
        logger.info(f"ChunkTextNode: Preparing to chunk {len(essays)} essays")
        # For each essay, create a tuple (essay_item, metadata)
        return [(essay_item, metadata) for essay_item in essays.items()]
    
    def exec(self, inputs):
        essay_item, metadata_df = inputs
        essay_id, text = essay_item
        logger.debug(f"ChunkTextNode: Chunking essay {essay_id} ({len(text)} chars)")
        # Get metadata for this essay
        essay_metadata = metadata_df[metadata_df['text_id'] == essay_id].iloc[0].to_dict() if not metadata_df[metadata_df['text_id'] == essay_id].empty else {}
        
        # Create chunks with identifiers
        chunks = []
        text_chunks = chunk_text(text)
        for i, chunk_content in enumerate(text_chunks):
            chunk_id = f"{essay_id}_chunk_{i}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_content,
                "essay_id": essay_id,
                "position": i,
                "title": essay_metadata.get("title", ""),
                "date": essay_metadata.get("date", "")
            })
        logger.debug(f"ChunkTextNode: Essay {essay_id} split into {len(chunks)} chunks")
        return chunks
    
    def post(self, shared, prep_res, exec_res_list):
        # Flatten the list of lists into a single list of chunks
        chunks = [chunk for chunks_list in exec_res_list for chunk in chunks_list]
        shared["chunks"] = chunks
        logger.info(f"ChunkTextNode: Created a total of {len(chunks)} chunks from all essays")
        return "default"


class GenerateEmbeddingsNode(BatchNode):
    def prep(self, shared):
        logger.info(f"GenerateEmbeddingsNode: Preparing to generate embeddings for {len(shared['chunks'])} chunks")
        return shared["chunks"]
    
    def exec(self, chunk):
        logger.debug(f"GenerateEmbeddingsNode: Generating embedding for chunk {chunk['id']}")
        embedding = get_embedding(chunk["text"])
        return chunk["id"], embedding, chunk
    
    def post(self, shared, prep_res, exec_res_list):
        chunk_ids = [result[0] for result in exec_res_list]
        embeddings = [result[1] for result in exec_res_list]
        chunks = [result[2] for result in exec_res_list]
        
        shared["embeddings"] = np.array(embeddings)
        shared["chunk_metadata"] = chunks
        logger.info(f"GenerateEmbeddingsNode: Generated {len(embeddings)} embeddings")
        return "default"


class StoreIndexNode(Node):
    def prep(self, shared):
        logger.info("StoreIndexNode: Preparing to create and store FAISS index")
        return shared["embeddings"], shared["faiss_index_path"]
    
    def exec(self, inputs):
        embeddings, index_path = inputs
        logger.info(f"StoreIndexNode: Creating FAISS index with {len(embeddings)} embeddings")
        index = create_index(embeddings)
        logger.info(f"StoreIndexNode: Saving index to {index_path}")
        save_index(index, index_path)
        return index
    
    def post(self, shared, prep_res, exec_res):
        shared["faiss_index"] = exec_res
        logger.info("StoreIndexNode: FAISS index created and stored successfully")
        return "default"


class StoreMetadataNode(Node):
    def prep(self, shared):
        logger.info(f"StoreMetadataNode: Preparing to store metadata for {len(shared['chunk_metadata'])} chunks")
        return shared["chunk_metadata"], shared["metadata_path"]
    
    def exec(self, inputs):
        chunk_metadata, metadata_path = inputs
        # Make sure directory exists
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        logger.info(f"StoreMetadataNode: Writing metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(chunk_metadata, f)
        return True
    
    def post(self, shared, prep_res, exec_res):
        logger.info(f"StoreMetadataNode: Metadata stored successfully: {exec_res}")
        return "default"


# -------------------- Online Processing Nodes --------------------

class VerifyQueryNode(Node):
    def prep(self, shared):
        logger.info("VerifyQueryNode: Verifying if query is relevant to Paul Graham's writings")
        return shared["query"]
    
    def exec(self, query):
        logger.info(f"VerifyQueryNode: Checking query: {query}")
        
        prompt = f"""Determine if the following user query is related to topics Paul Graham writes about, such as startups, entrepreneurship, programming, technology, YCombinator, or personal development.

User query: "{query}"

Return your analysis in YAML format:
```yaml
is_valid: true/false # true if the query is related to Paul Graham topics, false otherwise
reason: "Brief explanation of your decision"
```"""
        
        response = call_llm(prompt)
        
        # Extract YAML content
        yaml_content = response
        if "```yaml" in response:
            yaml_content = response.split("```yaml")[1].split("```")[0].strip()
        elif "```" in response:
            yaml_content = response.split("```")[1].strip()
        
        structured_result = yaml.safe_load(yaml_content)
        
        # Validate with assertions
        assert "is_valid" in structured_result, "Missing is_valid field"
        assert isinstance(structured_result["is_valid"], bool), "is_valid must be boolean"
        assert "reason" in structured_result, "Missing reason field"
        
        is_valid = structured_result["is_valid"] 
        reason = structured_result["reason"]
        
        logger.info(f"VerifyQueryNode: Query validity: {is_valid}, Reason: {reason}")
        return is_valid, reason
    
    def post(self, shared, prep_res, exec_res):
        is_valid, reason = exec_res
        shared["is_valid_query"] = is_valid
        shared["rejection_reason"] = reason if not is_valid else None
        
        if not is_valid:
            # Set a rejection message in Paul Graham's style
            shared["final_response"] = f"This is not a question I want to answer. Humm. I only advise on startup, programming, and entrepreneurship. Humm."
            logger.info(f"VerifyQueryNode: Query rejected: {reason}")
            return "invalid"
        
        logger.info("VerifyQueryNode: Query accepted, proceeding to processing")
        return "valid"

class ProcessQueryNode(Node):
    def prep(self, shared):
        logger.info(f"ProcessQueryNode: Processing query: '{shared['query']}'")
        return shared["query"]
    
    def exec(self, query):
        logger.info("ProcessQueryNode: Generating embedding for query")
        query_embedding = get_embedding(query)
        return query_embedding
    
    def post(self, shared, prep_res, exec_res):
        shared["query_embedding"] = exec_res
        logger.info("ProcessQueryNode: Query embedding generated and stored")
        return "default"


class RetrieveChunksNode(Node):
    def prep(self, shared):
        logger.info("RetrieveChunksNode: Preparing to retrieve relevant chunks")
        return shared["query_embedding"], shared["faiss_index"], shared["chunk_metadata"]
    
    def exec(self, inputs):
        query_embedding, faiss_index, chunk_metadata = inputs
        # Get top 5 matches
        logger.info("RetrieveChunksNode: Searching FAISS index for top 5 matches")
        scores, indices = search_index(query_embedding, faiss_index, top_k=5)
        
        # Retrieve the corresponding chunks
        retrieved_chunks = []
        for idx, score in zip(indices, scores):
            idx_int = int(idx)  # Convert numpy int to Python int
            metadata = chunk_metadata[idx_int]
            retrieved_chunks.append({
                "text": metadata["text"],
                "metadata": {k: v for k, v in metadata.items() if k != "text"},
                "score": float(score)  # Convert numpy float to Python float
            })
        
        logger.info(f"RetrieveChunksNode: Retrieved {len(retrieved_chunks)} chunks")
        return retrieved_chunks
    
    def post(self, shared, prep_res, exec_res):
        shared["retrieved_chunks"] = exec_res
        logger.info(f"RetrieveChunksNode: Stored {len(exec_res)} retrieved chunks")
        return "default"


class EvaluateChunksNode(BatchNode):
    def prep(self, shared):
        logger.info(f"EvaluateChunksNode: Preparing to evaluate {len(shared['retrieved_chunks'])} chunks")
        query = shared["query"]
        # For each chunk, create a tuple (chunk, query)
        return [(chunk, query) for chunk in shared["retrieved_chunks"]]
    
    def exec(self, inputs):
        chunk, query = inputs
        logger.info(f"EvaluateChunksNode: Evaluating chunk: '{chunk['text'][:100]}...' with similarity score {chunk['score']}")
        
        # Evaluate chunk relevance using LLM with YAML output
        prompt = f"""
Query: "{query}"

Text passage: "{chunk['text']}"

Evaluate if this passage is RELEVANT to answering the query.

For a passage to be relevant, it should contain information that directly helps answer the query 
or provides important context/background that would make the answer more complete and accurate.

Respond with a YAML structure that indicates:
- Whether the passage is relevant ("true" or "false")
- A brief explanation of your reasoning (1-2 sentences)

Output in YAML format:
```yaml
relevant: true/false
explanation: |
    your briefexplanation here
```"""
        
        response = call_llm(prompt)

        # Extract YAML content
        yaml_str = response.split("```yaml")[1].split("```")[0].strip() if "```yaml" in response else response
        result = yaml.safe_load(yaml_str)
        
        # Validate structure
        assert isinstance(result, dict)
        assert "relevant" in result
        assert "explanation" in result
        
        # Convert to boolean - accept both boolean values and string representations
        relevant_value = result["relevant"]
        if isinstance(relevant_value, bool):
            is_relevant = relevant_value
        else:
            # Handle string representations like "true", "false", etc.
            is_relevant = str(relevant_value).lower() in ["true", "yes", "1", "t", "y"]
            
        relevance_explanation = result["explanation"]
        
        # Add relevance info to chunk
        chunk["is_relevant"] = is_relevant
        chunk["relevance_explanation"] = relevance_explanation
        
        return chunk
    
    def post(self, shared, prep_res, exec_res_list):
        # Filter only relevant chunks
        relevant_chunks = [chunk for chunk in exec_res_list if chunk["is_relevant"]]
        shared["relevant_chunks"] = relevant_chunks
        
        logger.info(f"EvaluateChunksNode: {len(relevant_chunks)} out of {len(exec_res_list)} chunks determined to be relevant")
        return "default"


class SynthesizeResponseNode(Node):
    def prep(self, shared):
        logger.info(f"SynthesizeResponseNode: Preparing to synthesize response with {len(shared['relevant_chunks'])} relevant chunks")
        return shared["query"], shared["relevant_chunks"]
    
    def exec(self, inputs):
        query, chunks = inputs
        
        if not chunks:
            logger.warning("SynthesizeResponseNode: No relevant chunks found for this query")
            return {
                "metadata": {
                    "source_count": 0
                },
                "content": "I don't have enough information from my essays to answer this question confidently."
            }
        
        # Format chunks for prompt
        formatted_chunks = ""
        for i, chunk in enumerate(chunks):
            formatted_chunks += f"\n{i+1}\n{chunk['text']}\n"
            if "explanation" in chunk and chunk["explanation"]:
                formatted_chunks += f"EXPLANATION: {chunk['explanation']}\n"
        
        # Create prompt for LLM with YAML structure
        prompt = f"""
You are Paul Graham. 
1. Answer the following question in concise and to the point manner in 50 words or less, based ONLY on your knowledge.
2. If your knowledge base can't directly answer the question, say you are unsure.
   DON'T say "based on the provided knowledge base". That sounds like a robot. Say "based on my knowledge".
3. Be specific, and quote numbers if you can. E.g., instead of say "YC gave startup a lot of money", say "YC gave startup $500,000".

Paul Graham Humm a lot. Add Humm after you finish each point. (Humm not Hmm)

QUESTION: {query}

YOUR KNOWLEDGE BASE:
{formatted_chunks}

Please provide your response in 50 words or less, in the following YAML format:

```yaml
content: | 
    You need three things to create a successful startup. Humm.
    To start with good people, to make something customers actually want, and to spend as little money as possible. Humm.
    Most startups that fail do it because they fail at one of these. A startup that does all three will probably succeed. Humm.
```
"""
        
        logger.info("SynthesizeResponseNode: Calling LLM to generate final response")
        response = call_llm(prompt)

        # Extract YAML content
        yaml_str = response.split("```yaml")[1].split("```")[0].strip() if "```yaml" in response else response
        result = yaml.safe_load(yaml_str)
        
        # Validate structure
        assert isinstance(result, dict)
        
        # Check if result has content directly or needs additional extraction
        if "content" in result:
            # Already has the right structure
            structured_response = result
            # Ensure it has metadata
            if "metadata" not in structured_response:
                structured_response["metadata"] = {
                    "source_count": len(chunks)
                }
        else:
            # No content key found - use the whole result as content
            structured_response = {
                "metadata": {
                    "source_count": len(chunks)
                },
                "content": str(result)  # Convert the whole result to a string as content
            }
        
        return structured_response

    def post(self, shared, prep_res, exec_res):
        shared["final_response"] = exec_res
        content_preview = exec_res["content"][:50] + "..." if len(exec_res["content"]) > 50 else exec_res["content"]
        logger.info(f"SynthesizeResponseNode: Generated response with source count: {exec_res['metadata'].get('source_count', 0)}")
        logger.debug(f"SynthesizeResponseNode: Response preview: {content_preview}")
        return "default"


class TextToSpeechNode(Node):
    def prep(self, shared):
        logger.info("TextToSpeechNode: Preparing to convert text to speech")
        return shared["final_response"]
    
    def exec(self, response):
        logger.info("TextToSpeechNode: Converting response to speech")
        # Extract the content from the structured response
        if isinstance(response, dict) and "content" in response:
            logger.debug("TextToSpeechNode: Using structured response content")
            content = response["content"]
        else:
            logger.warning("TextToSpeechNode: Response is not structured, using as is")
            content = response
            
        audio_file_hash = synthesize_text_to_speech(content)
        return audio_file_hash
    
    def post(self, shared, prep_res, exec_res):
        shared["audio_file_hash"] = exec_res
        logger.info(f"TextToSpeechNode: Audio file generated with hash {exec_res}")
        return "default"


# -------------------- Flow Definitions --------------------

# Offline Processing Flow
load_essays = LoadEssaysNode()
chunk_text_node = ChunkTextNode()
generate_embeddings = GenerateEmbeddingsNode()
store_index = StoreIndexNode()
store_metadata = StoreMetadataNode()

# Connect nodes
load_essays >> chunk_text_node >> generate_embeddings >> store_index >> store_metadata

# Create flow
offline_flow = Flow(start=load_essays)
logger.info("Offline processing flow initialized")

# Online Processing Flow
verify_query = VerifyQueryNode()
process_query = ProcessQueryNode()
retrieve_chunks = RetrieveChunksNode()
evaluate_chunks = EvaluateChunksNode()
synthesize_response = SynthesizeResponseNode()
text_to_speech = TextToSpeechNode()

# Connect nodes
verify_query - "valid" >> process_query
verify_query - "invalid" >> text_to_speech
process_query >> retrieve_chunks >> evaluate_chunks >> synthesize_response >> text_to_speech

# Create flow
online_flow = Flow(start=verify_query)
logger.info("Online processing flow initialized")