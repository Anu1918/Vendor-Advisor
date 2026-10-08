# Vendor Selection Advisor (Use case #6)

Streamlit app: upload vendors, set weights/constraints, get a ranked shortlist, an AI-written rationale (Gemini), sanity checks and a stability test.

## Run locally
```
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key
streamlit run app.py
```
Free key: https://aistudio.google.com/apikey . Without a key the app still works and shows a template explanation.
If Gemini returns a model-not-found error, set `GEMINI_MODEL` (env var) to a current model name from ai.google.dev.

## Deploy a shareable link (free)
1. Push this folder to a public GitHub repo (do NOT commit `.streamlit/secrets.toml`; .gitignore already excludes it).
2. Go to share.streamlit.io, sign in with GitHub, New app, pick the repo, main file `app.py`.
3. Advanced settings > Secrets: `GEMINI_API_KEY = "your-key"`.
4. Deploy, then paste the URL into the project document.
