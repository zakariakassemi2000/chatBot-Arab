import os
import pickle
import time
from engine.retriever import FAISSRetriever

def main():
    print("Loading old dataframe...")
    old_data_path = "models/retriever_data.pkl"
    if not os.path.exists(old_data_path):
        print(f"File {old_data_path} not found. Attempting to load from datasets...")
        from data.knowledge_base import load_and_prepare_datasets
        df = load_and_prepare_datasets()
    else:
        with open(old_data_path, "rb") as f:
            data = pickle.load(f)
        df = data["df"]
        
    print(f"Loaded {len(df)} rows.")
    print("Rebuilding index with new Camel embedding...")
    
    r = FAISSRetriever()
    
    start_time = time.time()
    r.build_index(df, verbose=True)
    r.save()
    end_time = time.time()
    
    print(f"Rebuild completed successfully.")
    print(f"Rebuild time: {end_time - start_time:.2f} seconds")
    
if __name__ == '__main__':
    main()
