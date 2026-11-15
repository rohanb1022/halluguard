import sys
import webbrowser
import threading
import time
from app import app, kb

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    print("=" * 70)
    print("  🛡️  HalluGuard: Claim-Level Hallucination Detector & Guard")
    print("  Authors: Suhas Arali, Aditya Gaikar, Rohan Bhangale, Ayush Karath")
    print("  Faculty Guide: Prof. Rakhi Jadhav | Generative AI PoC")
    print("=" * 70)
    print(f"  [✓] Loaded {len(kb.sentences)} knowledge base facts across {len(kb.file_contents)} files.")
    print("  [✓] Ready at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 70)

    # Launch browser automatically
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host="127.0.0.1", port=5000, debug=False)

