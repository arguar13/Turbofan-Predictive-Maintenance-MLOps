import os

# main.py refuses to import without API_KEY. Set before any test imports it;
# setdefault keeps a value exported from outside.
os.environ.setdefault("API_KEY", "test-api-key")
