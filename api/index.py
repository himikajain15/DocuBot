from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        html = """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>DocuBot Deployment Status</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }
      .card { max-width: 760px; border: 1px solid #ddd; border-radius: 10px; padding: 1.2rem; }
      code { background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 4px; }
    </style>
  </head>
  <body>
    <div class=\"card\">
      <h1>DocuBot is deployed</h1>
      <p>This repository contains a Streamlit app (<code>DocuBot.py</code>).</p>
      <p>Vercel serverless deployment does not run Streamlit directly as a website root process. That is why your original URL returned <code>404 NOT_FOUND</code>.</p>
      <p>Use Streamlit Community Cloud or Render/Railway for full Streamlit hosting, or migrate this app to FastAPI/Flask for native Vercel serverless deployment.</p>
    </div>
  </body>
</html>
"""
        self.wfile.write(html.encode("utf-8"))