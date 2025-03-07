import os
import argparse
from flow import offline_flow

def main(args):
    # Define paths
    data_dir = args.data_dir
    meta_csv = args.meta_csv
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define shared data structure for the flow
    shared = {
        # Input paths
        "data_dir": data_dir,
        "meta_csv": meta_csv,
        
        # Output paths
        "faiss_index_path": os.path.join(output_dir, "essay_index.faiss"),
        "metadata_path": os.path.join(output_dir, "chunk_metadata.json"),
    }
    
    print("Starting offline processing...")
    print(f"Data directory: {data_dir}")
    print(f"Metadata CSV: {meta_csv}")
    print(f"Output directory: {output_dir}")
    
    # Run the offline processing flow
    offline_flow.run(shared)
    
    print("\nOffline processing completed successfully!")
    print(f"FAISS index saved to: {shared['faiss_index_path']}")
    print(f"Chunk metadata saved to: {shared['metadata_path']}")
    print(f"Processed {len(shared['chunks'])} chunks from {len(shared['essays'])} essays")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Paul Graham essays for RAG")
    parser.add_argument("--data-dir", type=str, default="data", 
                        help="Directory containing the essay text files")
    parser.add_argument("--meta-csv", type=str, default="meta.csv",
                        help="Path to CSV file with essay metadata")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save processed files")
    
    args = parser.parse_args()
    main(args) 