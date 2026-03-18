# DocuBot

DocuBot is a Streamlit app that lets you chat with PDF files and website content using Groq and LangChain.

## Run locally

```bash
pip install -r requirements.txt
streamlit run DocuBot.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select this repository and branch.
4. Set the main file path to `DocuBot.py`.
5. In the app settings, add a secret:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

You can also copy [.streamlit/secrets.toml.example](/n:/Projects/Docubot/.streamlit/secrets.toml.example) when testing locally.

## Notes

- This project is better suited to Streamlit Community Cloud than Vercel because the UI is a Streamlit app, not a serverless API.
- The deployed app can accept either a website URL or a PDF upload.
