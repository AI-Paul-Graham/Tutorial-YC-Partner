import os
import json
import argparse
from flow import online_flow
from utils.vector_search import load_index

def initialize_system(config):
    """Load necessary resources at system startup."""
    print(f"Loading system resources from {config['output_dir']}...")
    
    # Load FAISS index
    faiss_index_path = os.path.join(config['output_dir'], "essay_index.faiss")
    faiss_index = load_index(faiss_index_path)
    print(f"Loaded FAISS index from {faiss_index_path}")
    
    # Load chunk metadata (includes text content)
    metadata_path = os.path.join(config['output_dir'], "chunk_metadata.json")
    with open(metadata_path, "r") as f:
        chunk_metadata = json.load(f)
    print(f"Loaded {len(chunk_metadata)} chunks from {metadata_path}")
        
    return {
        "faiss_index": faiss_index,
        "chunk_metadata": chunk_metadata
    }

def main(args):
    # Initialize system with resources
    system_resources = initialize_system({"output_dir": args.output_dir})
    
    # Process user queries in a loop
    while True:
        # Get user query
        query = input("\nAsk a question about Paul Graham's essays (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        # Create shared data for this query
        shared = {
            # System resources
            "faiss_index": system_resources["faiss_index"],
            "chunk_metadata": system_resources["chunk_metadata"],
            
            # Query
            "query": query
        }
        
        # Run the online processing flow
        online_flow.run(shared)
        
        # Display results
        print("\n" + "-" * 50)
        if not shared.get("is_valid_query", True):
            print(f"Query was determined to be off-topic. Reason: {shared.get('rejection_reason', 'Unknown')}")
        print(f"Paul Graham's response: \n\n{shared['final_response']}")
        print("-" * 50)
        print(f"Audio response available with hash: {shared['audio_file_hash']}")
        print(f"(Check the audio_cache directory for the MP3 file)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Paul Graham - Ask questions about Paul Graham's essays")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory where processed files are stored")
    
    args = parser.parse_args()
    main(args)