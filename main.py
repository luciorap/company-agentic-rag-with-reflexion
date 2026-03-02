import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    try:
        print("Hello Company Advanced Agent", flush=True)
        
        question = "What is the financial analysis?"
        print(f"Question: {question}", flush=True)
        
        result = app.invoke(input={
            "question": question,
            "generation": "",
            "web_search": False,
            "documents": [],
            "sources": [],
            "messages": [],
            "iteration_count": 0
        })
        
        print("\n---FINAL REPORT---", flush=True)
        gen = result.get("generation", "No generation available.")
        print(gen, flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
