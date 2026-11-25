import importlib
modules = ["streamlit", "fitz", "numpy", "sentence_transformers", "google.genai"]
for name in modules:
    importlib.import_module(name)
    print("OK:", name)
print("Environment imports passed.")
