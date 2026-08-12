# brd-srs-to-test-case

Research core for comparing controlled BRD/SRS-to-test-case generation conditions.

[Operations guide](docs/research-core-operations.md)

## Research comparison

```sh
env PYTHONPATH=src .venv/bin/python -m streamlit run app.py
```

## Existing prototypes

# Run with local Gemma 4

```sh
pip install streamlit pandas pypdf openpyxl
ollama serve
ollama pull gemma4
python -m streamlit run app-ba.py
```

Both prototypes support Gemini and local Gemma 4 through Ollama.
