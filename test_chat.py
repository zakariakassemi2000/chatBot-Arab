
import sys
import os

# Add current dir to sys.path
sys.path.append(os.getcwd())

from engine.retriever import FAISSRetriever
from engine.classifier import IntentClassifier

print("Testing Chatbot Components...")
try:
    r = FAISSRetriever()
    c = IntentClassifier()
    r_ok = r.load()
    c_ok = c.load()
    print(f"Retriever loaded: {r_ok}")
    print(f"Classifier loaded: {c_ok}")
    
    if r_ok:
        query = "عندي ألم في الرأس"
        res = r.get_best_answer(query)
        print(f"Query: {query}")
        print(f"Result: {res[0][:100]}...")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILURE: {e}")
