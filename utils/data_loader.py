import os
import pandas as pd

def load_data(data_dir, meta_csv):
    """Load essays and metadata from files.
    
    Args:
        data_dir (str): Directory containing the essay text files
        meta_csv (str): Path to the CSV file with metadata
        
    Returns:
        tuple: (essays_dict, metadata_dataframe)
            essays_dict: Dictionary mapping file_id to essay text
            metadata_dataframe: Pandas DataFrame with essay metadata
    """
    metadata = pd.read_csv(meta_csv)
    essays = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                file_id = filename.split('.')[0]
                essays[file_id] = f.read()
    
    return essays, metadata

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(current_dir)
    
    data_dir = os.path.join(project_root, "data")
    meta_csv = os.path.join(project_root, "meta.csv")
    
    if os.path.exists(data_dir) and os.path.exists(meta_csv):
        essays, metadata = load_data(data_dir, meta_csv)
        
        print(f"Loaded {len(essays)} essays")
        print(f"Metadata shape: {metadata.shape}")
        
        # Print a sample essay
        if essays:
            sample_id = next(iter(essays))
            print(f"\nSample essay (ID: {sample_id}):")
            print(f"{essays[sample_id][:200]}...")
    else:
        print(f"Data directory or metadata file not found.")
        print(f"Expected data directory: {data_dir}")
        print(f"Expected metadata file: {meta_csv}") 