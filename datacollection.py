"""
Backward-compatible wrapper for data collection.
Directly invokes data_collection.py with full crash-proofing and sign-switching.
"""
from data_collection import main

if __name__ == "__main__":
    main()