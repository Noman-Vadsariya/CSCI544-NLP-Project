#!/usr/bin/env python3
import threading
import html
import secrets
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
import markdown

HOST = "127.0.0.1"     # 🔒 loopback only
PORT = 7680

SESSION_TOKEN = secrets.token_urlsafe(32)  # 🔐 per-run secret
LATEST_RAW = ""
LOCK = threading.Lock()

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Secure LLM Render</title>

<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">

<link rel="stylesheet"
 href="https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css">

<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: false
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>


<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>

<style>
body {{
  background: #f6f8fa;
  padding: 24px;
  font-family: system-ui, sans-serif;
}}
.wrap {{
  max-width: 980px;
  margin: auto;
  background: #fff;
  padding: 28px;
  border-radius: 10px;
}}
</style>
</head>
<body>
<div class="wrap markdown-body">
{content}
</div>
<script>
setTimeout(() => location.reload(), 1000);
</script>
</body>
</html>
"""

# --- Math protection utilities ---------------------------------------------

MATH_PATTERNS = [
    r"\$\$.*?\$\$",           # $$ ... $$
    r"\\\[.*?\\\]",           # \[ ... \]
    r"\\\(.*?\\\)",           # \( ... \)
]

MATH_RE = re.compile("|".join(MATH_PATTERNS), re.DOTALL)

def extract_math(text):
    blocks = []
    def repl(match):
        blocks.append(match.group(0))
        return f"@@MATH{len(blocks)-1}@@"
    return MATH_RE.sub(repl, text), blocks

def restore_math(text, blocks):
    for i, block in enumerate(blocks):
        text = text.replace(f"@@MATH{i}@@", block)
    return text

def normalize_lists(text: str) -> str:
    """
    Ensure that any line starting with '- ' has a blank line above it,
    unless it is already preceded by a blank line or is at the start.
    """
    return re.sub(
        r"(?m)(?<!\n)\n(?=- )",
        "\n\n",
        text
    )


def safe_markdown(raw: str) -> str:
    """
    SECURITY + CORRECTNESS:
    - Extract math blocks verbatim
    - Escape HTML elsewhere (XSS safe)
    - Run markdown
    - Restore math untouched for MathJax
    """
    stripped, math_blocks = extract_math(raw)
    stripped = normalize_lists(stripped)
    escaped = html.escape(stripped)
    rendered = markdown.markdown(
        escaped,
        extensions=["extra", "tables", "fenced_code"]
    )
    return restore_math(rendered, math_blocks)

# --- HTTP handler -----------------------------------------------------------

class SecureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if f"token={SESSION_TOKEN}" not in self.path:
            self.send_response(403)
            self.end_headers()
            return

        with LOCK:
            raw = LATEST_RAW

        content = safe_markdown(raw) if raw.strip() else "<em>No input yet.</em>"
        page = HTML_TEMPLATE.format(content=content).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")

        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; "
            "style-src 'self' https://cdn.jsdelivr.net; "
            "script-src https://cdn.jsdelivr.net; "
            "font-src https://cdn.jsdelivr.net"
        )

        self.end_headers()
        self.wfile.write(page)

    def log_message(self, *args):
        return  # 🔇 silence

# --- stdin reader -----------------------------------------------------------

import sys
import select
import time

def stdin_reader():
    global LATEST_RAW

    print("\n🔐 SECURE LLM RENDERER")
    print(f"🌐 URL:")
    print(f"   http://{HOST}:{PORT}/?token={SESSION_TOKEN}\n")
    print("📥 Paste proprietary text here.")
    print("   Commands: :clear  :quit\n")

    buf = []
    paste_buf = []
    last_input_time = None
    PASTE_GAP = 0.15  # seconds; tweak if needed

    while True:
        # wait for stdin with timeout
        r, _, _ = select.select([sys.stdin], [], [], PASTE_GAP)

        if r:
            line = sys.stdin.readline()
            if not line:
                return

            line = line.rstrip("\n")
            paste_buf.append(line)
            last_input_time = time.time()
        else:
            # timeout hit — if we have buffered paste data, commit it
            if paste_buf:
                # join pasted text as ONE logical piece
                joined = "\n".join(paste_buf)

                if joined.strip() == ":quit":
                    import os
                    os._exit(0)

                if joined.strip() == ":clear":
                    buf = []
                    with LOCK:
                        LATEST_RAW = ""
                else:
                    buf.append(joined)
                    with LOCK:
                        LATEST_RAW = "\n" + joined + "\n"

                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                paste_buf = []

# --- main ------------------------------------------------------------------

def main():
    threading.Thread(target=stdin_reader, daemon=True).start()
    server = HTTPServer((HOST, PORT), SecureHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
