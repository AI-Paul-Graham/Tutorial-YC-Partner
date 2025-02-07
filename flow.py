import numpy as np
from typing import Dict, List
import yaml
import re
import os
import pandas as pd
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from pocketflow import *

def call_llm(prompt):
    from anthropic import AnthropicVertex
    client = AnthropicVertex(region="us-east5", project_id="wu-lab")
    response = client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        model="claude-3-5-sonnet-v2@20241022"
    )
    return response.content[0].text

class ValidateAndScreenQuestion(Node):
    def prep(self, shared):
        question = shared.get("question", "")
        # Crop to first 5000 characters
        cropped = question[:5000]
        shared["cropped_question"] = cropped
        return cropped
        
    def exec(self, question):
        prompt = f"""
Given this question: {question}

First determine if this is appropriate for a YC partner to answer. Consider:
1. Is it related to startups, technology, business, or entrepreneurship?
2. Is it appropriate and professional?

If NOT appropriate, generate a polite response explaining why we cannot answer.
If appropriate, just return that it's valid.

Output in yaml:
```yaml
is_valid: true/false
response: if not valid, politely say why you can't answer
```
"""
        resp = call_llm(prompt)
        yaml_str = resp.split("```yaml")[1].split("```")[0].strip()
        result = yaml.safe_load(yaml_str)
        
        # Validate response
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert isinstance(result["is_valid"], bool)
        if not result["is_valid"]:
            assert "response" in result
            
        return result

    def post(self, shared, prep_res, exec_res):
        if not exec_res["is_valid"]:
            shared["final_answer"] = {
                "citations": [],
                "summary": exec_res["response"]
            }
            return "invalid"
        return "valid"

class GetRelevantChunks(Node):
    def prep(self, shared):
        return (
            shared["cropped_question"],
            shared["embeddings_df"],
            shared.get("chunk_texts", {})
        )
        
    def exec(self, inputs):
        question, embeddings_df, chunk_texts = inputs
        
        # Get question embedding
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        question_input = TextEmbeddingInput(question, "RETRIEVAL_QUERY")
        question_embedding = model.get_embeddings([question_input])[0]
        
        # Convert question embedding to numpy array
        question_vector = np.array(question_embedding.values)
        
        # Get all chunk embeddings as a matrix
        chunk_embeddings = np.stack(embeddings_df['embedding'].values)
        
        # Compute cosine similarity
        similarities = np.dot(chunk_embeddings, question_vector) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_vector)
        )
        
        # Get top 10 chunks
        top_indices = np.argsort(similarities)[-10:][::-1]
        
        results = []
        for idx in top_indices:
            row = embeddings_df.iloc[idx]
            text_id, chunk_id = row['text_id'], row['chunk_id']
            
            # Get chunk text and expand it
            chunk_text = chunk_texts.get((text_id, chunk_id), "")
            chunk_start = max(0, chunk_id * 10000 - 2000)
            chunk_end = min(len(chunk_text), (chunk_id + 1) * 10000 + 2000)
            expanded_text = chunk_text[chunk_start:chunk_end]
            
            results.append({
                'text_id': text_id,
                'chunk_id': chunk_id,
                'score': float(similarities[idx]),
                'content': chunk_text,
                'expanded_content': expanded_text
            })
            
        return results

    def post(self, shared, prep_res, exec_res):
        shared["relevant_chunks"] = exec_res
            
class ComposeAnswer(Node):
    def prep(self, shared):
        return (
            shared["cropped_question"],
            shared["relevant_chunks"],
            shared["meta_df"]
        )
        
    def exec(self, inputs):
        question, chunks, meta_df = inputs
        
        # Format chunks for the prompt with local IDs
        formatted_chunks = []
        chunk_id_map = {}  # Map local_id to text_id
        
        for i, chunk in enumerate(chunks, 1):
            text_id = chunk['text_id']
            chunk_id_map[i] = text_id
            meta_row = meta_df[meta_df['text_id'] == text_id].iloc[0]
            formatted_chunks.append(f"""
Source {i}: {meta_row['title']} ({meta_row['link']})
Content: {chunk['expanded_content']}
""")
        
        prompt = f"""
Question: {question}

Here are relevant sources:
{formatted_chunks}

Create a response with citations and summary. For citations:
1. Use source numbers (1-{len(chunks)}) to refer to sources
2. Keep the original text but you can:
   - Fix typos, capitalization and punctuation
   - Skip irrelevant parts using [...]
   - Remove formatting artifacts
3. Each citation should be a direct quote from the source

Example citation format:
- Good: "Founders should focus on product-market fit [...]  and then scale"
- Good: "Startups need to be 'default alive' [meaning they survive on existing revenue]"
- Bad: paraphrasing or changing the original text meaning

Output in yaml:
```yaml
citations:
  - source_id: local source number (1-{len(chunks)})
    citation: cleaned up quote from that source
summary: comprehensive answer
```
"""
        resp = call_llm(prompt)
        yaml_str = resp.split("```yaml")[1].split("```")[0].strip()
        result = yaml.safe_load(yaml_str)
        
        # Validate
        assert isinstance(result, dict)
        assert "citations" in result
        assert "summary" in result
        
        # Convert local source_ids to text_ids
        for citation in result["citations"]:
            local_id = citation.pop("source_id")
            citation["text_id"] = chunk_id_map[local_id]
        
        return result

    def post(self, shared, prep_res, exec_res):
        shared["final_answer"] = exec_res

# Connect nodes
validate = ValidateAndScreenQuestion(max_retries=3,wait=30)
# validate = ValidateAndScreenQuestion()
get_chunks = GetRelevantChunks()
compose = ComposeAnswer(max_retries=3,wait=30)
# compose = ComposeAnswer()

validate - "valid" >> get_chunks >> compose
# "invalid" path just ends as the response is already set

# Create flow
partner_flow = Flow(start=validate)

