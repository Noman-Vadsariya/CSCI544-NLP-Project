#!/usr/bin/env python3
"""
Secure Localhost Markdown Chat Renderer (ChatGPT-like UI)

- Binds to 127.0.0.1 only (loopback)
- Per-run secret token required on ALL routes (HTML, JS, CSS, API)
- Accepts continuous text submissions:
  - From terminal (paste like your original script)
  - (Optional) From the browser input box
- Renders a chat transcript where:
  - User bubble = raw text (escaped)
  - "Formatter bot" bubble = nicely rendered markdown (XSS-safe) + MathJax

Commands (terminal or browser input):
  :clear   clears chat
  :quit    exits immediately
"""

# --- Check if HF_HUB_OFFLINE is set -----------------------------------------------------------

import os
import sys

if os.environ.get("HF_HUB_OFFLINE") != "1":
    print(
        "ERROR: HF_HUB_OFFLINE is not set to '1'.\n"
        "Please copy this command (using ctrl+Ins), then run it:\n\n"
        "export HF_HUB_OFFLINE=1 ; python qwen25_coder_chat_html.py\n\n"
        # "  export HF_HUB_OFFLINE=1   (Linux/macOS)\n"
        # "  set HF_HUB_OFFLINE=1      (Windows CMD)\n"
        # "  $env:HF_HUB_OFFLINE=1     (PowerShell)"
    )
    sys.exit(1)

# --- Imports -----------------------------------------------------------

import html
import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import markdown
import select
import time

from qwen25_coder_chat_terminal import *
bot = ChatBot()

# --- Global state -----------------------------------------------------------

HOST = "127.0.0.1"  # 🔒 loopback only
PORT = 7681

SESSION_TOKEN = secrets.token_urlsafe(32)  # 🔐 per-run secret
LOCK = threading.Lock()

# Each item: {"id": int, "user_raw": str, "assistant_html": str}
MESSAGES = []
NEXT_ID = 1

# --- Math protection utilities ---------------------------------------------

MATH_PATTERNS = [
    r"\$\$.*?\$\$",     # display math
    r"\\\(.*?\\\)",     # inline math
]


MATH_RE = re.compile("|".join(MATH_PATTERNS), re.DOTALL)

# --- Markdown rendering with math protection -------------------------------

def extract_math(text: str):
    blocks = []

    def repl(match):
        blocks.append(match.group(0))
        return f"@@MATH{len(blocks)-1}@@"

    return MATH_RE.sub(repl, text), blocks

def restore_math(text: str, blocks):
    for i, block in enumerate(blocks):
        text = text.replace(f"@@MATH{i}@@", block)
    return text

def normalize_lists(text: str) -> str:
    # Ensure blank line before "- " items (helps markdown render pasted blocks)
    return re.sub(r"(?m)(?<!\n)\n(?=- )", "\n\n", text)

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
        extensions=["extra", "tables", "fenced_code"],
        output_format="html5",
    )
    return restore_math(rendered, math_blocks)


# --- App UI (HTML/CSS/JS) ---------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Secure Markdown Chat</title>

  <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css">
  <link rel="stylesheet" href="/app.css?token={token}">
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="title">
        <div class="dot"></div>
        <div>
          <div class="h1">Ayu-GPT</div>
          <div class="h2">Highly Secure • Completely Propietary • Go Blue!</div>
        </div>
      </div>
    </header>

    <main class="chat" id="chat">
      <div class="empty" id="emptyState">
        <div class="emptyCard">
          <div class="emptyTitle">
            Start chatting by entering text in the chatbox below. Press Enter once to submit.
          </div>
          <div class="emptySub">
            Please note: the chatbot may take some time to provide a response. Please wait patiently, and avoid re-submitting queries.
          </div>
          <div class="emptyHint">
            Terminal commands:
            <code>:clear</code> to clear the chat.
            <code>:quit</code> to terminate the program.
          </div>
        </div>
      </div>
      <div class="messages" id="messages"></div>
    </main>

    <footer class="composer">
      <textarea id="input" placeholder="Type your message here..."></textarea>
      <button id="sendBtn" class="btn primary">Send</button>
    </footer>
  </div>

  <script src="/app.js?token={token}"></script>
</body>
</html>
"""

APP_CSS = r"""
:root{
  --bg:#0b0f19;
  --panel:#0f172a;
  --panel2:#111c33;
  --border:rgba(255,255,255,0.08);
  --text:rgba(255,255,255,0.92);
  --muted:rgba(255,255,255,0.62);
  --accent:#7c3aed;
  --accent2:#a78bfa;
  --user:#1f2937;
  --assistant:#0b1222;
  --shadow: 0 10px 30px rgba(0,0,0,0.35);
}

*{ box-sizing:border-box; }
html,body{ height:100%; }
body{
  margin:0;
  background: radial-gradient(1200px 800px at 30% -10%, rgba(124,58,237,0.30), transparent 60%),
              radial-gradient(900px 600px at 90% 0%, rgba(167,139,250,0.18), transparent 55%),
              var(--bg);
  color:var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}

.shell{
  height:100%;
  display:flex;
  flex-direction:column;
  max-width: 1100px;
  margin: 0 auto;
}

.topbar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding: 18px 18px 12px 18px;
}

.title{
  display:flex;
  align-items:center;
  gap:12px;
}
.dot{
  width:12px;height:12px;border-radius:999px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  box-shadow: 0 0 0 6px rgba(124,58,237,0.15);
}
.h1{ font-weight:700; letter-spacing:0.2px; }
.h2{ font-size:12px; color:var(--muted); margin-top:2px; }

.actions{ display:flex; gap:10px; }

.btn{
  border:1px solid var(--border);
  background: rgba(255,255,255,0.04);
  color: var(--text);
  padding: 9px 12px;
  border-radius: 10px;
  cursor:pointer;
  transition: transform .05s ease, background .15s ease;
}
.btn:hover{ background: rgba(255,255,255,0.07); }
.btn:active{ transform: translateY(1px); }
.btn.primary{
  border-color: rgba(124,58,237,0.55);
  background: rgba(124,58,237,0.22);
}
.btn.primary:hover{ background: rgba(124,58,237,0.28); }
.btn.ghost{
  background: transparent;
}

.chat{
  flex:1;
  padding: 0 18px 12px 18px;
  overflow:auto;
}

.messages{
  display:flex;
  flex-direction:column;
  gap: 14px;
  padding-bottom: 16px;
}

.empty{
  display:block;
  padding: 18px 0;
}
.emptyCard{
  border: 1px dashed rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 18px;
}
.emptyTitle{ font-weight:700; margin-bottom:6px; }
.emptySub{ color: var(--muted); font-size: 13px; line-height:1.35; }
.emptyHint{ margin-top:10px; color: var(--muted); font-size: 12px; }
.emptyHint code{
  background: rgba(255,255,255,0.07);
  border:1px solid var(--border);
  border-radius: 8px;
  padding: 2px 6px;
}

.row{
  display:flex;
  width:100%;
}
.row.user{ justify-content:flex-end; }
.row.assistant{ justify-content:flex-start; }

.bubble{
  max-width: 860px;
  width: fit-content;
  border-radius: 16px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  overflow:hidden;
}
.bubble.user{
  background: rgba(31,41,55,0.55);
}
.bubble.assistant{
  background: rgba(15,23,42,0.72);
}

.meta{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  padding: 10px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  color: var(--muted);
  font-size: 12px;
}
.who{
  display:flex;
  align-items:center;
  gap:8px;
}
.badge{
  font-size: 11px;
  border:1px solid rgba(255,255,255,0.12);
  padding: 2px 8px;
  border-radius: 999px;
  color: rgba(255,255,255,0.75);
}

.content{
  padding: 12px 14px;
  font-size: 14px;
  line-height: 1.45;
}

/* user raw */
.user pre{
  margin:0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 13px;
  color: rgba(255,255,255,0.92);
}

/* assistant markdown */
.assistant .markdown-body{
  background: transparent;
  color: rgba(255,255,255,0.92);
  font-size: 14px;
}

/* --- FIX: force light code text on dark background --- */
.assistant .markdown-body pre,
.assistant .markdown-body code {
  color: rgba(255,255,255,0.92) !important;
}

.assistant .markdown-body pre code {
  color: inherit !important;
}

.assistant .markdown-body table{
  display:block;
  overflow:auto;
  max-width: 100%;
}

.composer{
  padding: 12px 18px 18px 18px;
  display:flex;
  gap: 10px;
  align-items:flex-end;
}
.composer textarea{
  flex:1;
  min-height: 44px;
  max-height: 180px;
  resize: vertical;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  color: var(--text);
  outline: none;
}
.composer textarea:focus{
  border-color: rgba(167,139,250,0.55);
  box-shadow: 0 0 0 6px rgba(124,58,237,0.12);
}
"""

APP_JS = r"""
(() => {
  const token = new URLSearchParams(location.search).get("token");
  const chat = document.getElementById("chat");
  const msgsEl = document.getElementById("messages");
  const emptyState = document.getElementById("emptyState");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");

  let rendered = new Set();
  let lastId = 0;

  function escapeHtml(s){
    return s.replace(/[&<>"']/g, c => ({
      "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
    }[c]));
  }

  function ensureMathJax(){
    if (window.__mathjax_loaded) return;
    window.__mathjax_loaded = true;

    window.MathJax = {
      tex: {
        inlineMath: [['\\(', '\\)']],
        displayMath: [['$$', '$$']],
        processEscapes: false
      },
      svg: { fontCache: 'global' }
    };


    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js";
    s.async = true;
    document.head.appendChild(s);
  }

  function scrollToBottomIfNear(){
    const threshold = 160;
    const nearBottom = chat.scrollHeight - (chat.scrollTop + chat.clientHeight) < threshold;
    if (nearBottom) chat.scrollTop = chat.scrollHeight;
  }

  function renderMessage(m){
    // User row
    const userRow = document.createElement("div");
    userRow.className = "row user";
    userRow.dataset.mid = String(m.id);

    const userBubble = document.createElement("div");
    userBubble.className = "bubble user";

    const userMeta = document.createElement("div");
    userMeta.className = "meta";
    userMeta.innerHTML = `
      <div class="who"><span class="badge">You</span></div>
      <div>#${m.id}</div>
    `;

    const userContent = document.createElement("div");
    userContent.className = "content user";
    userContent.innerHTML = `<pre>${escapeHtml(m.user_raw || "")}</pre>`;

    userBubble.appendChild(userMeta);
    userBubble.appendChild(userContent);
    userRow.appendChild(userBubble);

    // Assistant row
    const botRow = document.createElement("div");
    botRow.className = "row assistant";
    botRow.dataset.mid = String(m.id);

    const botBubble = document.createElement("div");
    botBubble.className = "bubble assistant";

    const botMeta = document.createElement("div");
    botMeta.className = "meta";
    botMeta.innerHTML = `
      <div class="who"><span class="badge">Formatter</span></div>
      <div>#${m.id}</div>
    `;

    const botContent = document.createElement("div");
    botContent.className = "content assistant";
    botContent.innerHTML = `<div class="markdown-body">${m.assistant_html || ""}</div>`;

    botBubble.appendChild(botMeta);
    botBubble.appendChild(botContent);
    botRow.appendChild(botBubble);

    msgsEl.appendChild(userRow);
    msgsEl.appendChild(botRow);
  }

  async function fetchMessages(){
    const r = await fetch(`/api/messages?token=${encodeURIComponent(token)}&after_id=${lastId}`, {
      cache: "no-store"
    });
    if (!r.ok) return;
    const data = await r.json();
    const items = data.messages || [];
    if (items.length === 0) return;

    ensureMathJax();

    emptyState.style.display = "none";
    for (const m of items){
      if (rendered.has(m.id)) continue;

      renderMessage(m);
      rendered.add(m.id);
      lastId = Math.max(lastId, m.id);
    }

    // typeset math
    if (window.MathJax && window.MathJax.typesetPromise){
      try { await window.MathJax.typesetPromise(); } catch(e) {}
    }

    scrollToBottomIfNear();
  }

  async function submit(text){
    const body = JSON.stringify({ text });
    const r = await fetch(`/api/submit?token=${encodeURIComponent(token)}`, {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body
    });
    if (!r.ok) return;
    input.value = "";
    // fetch immediately after submit
    await fetchMessages();
    chat.scrollTop = chat.scrollHeight;
  }

  sendBtn.addEventListener("click", () => {
    const t = (input.value || "").trimEnd();
    if (!t) return;
    submit(t);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();   // stop newline
      const t = (input.value || "").trimEnd();
      if (!t) return;
      submit(t);
    }
  });


  // Poll every 900ms (snappy but light)
  fetchMessages();
  setInterval(fetchMessages, 900);
})();
"""


# --- Helpers ----------------------------------------------------------------

def _get_token_from_query(path: str) -> str | None:
    q = parse_qs(urlparse(path).query)
    t = q.get("token", [None])[0]
    return t

def _require_token(handler: BaseHTTPRequestHandler) -> bool:
    t = _get_token_from_query(handler.path)
    if t != SESSION_TOKEN:
        handler.send_response(403)
        handler.send_header("Content-Type", "text/plain; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(b"Forbidden\n")
        return False
    return True

def _send_common_headers(handler: BaseHTTPRequestHandler, content_type: str):
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("X-Frame-Options", "DENY")
    # Tight CSP, but we allow:
    # - self for app.js/app.css and API fetches
    # - jsdelivr for github-markdown-css and MathJax
    handler.send_header(
        "Content-Security-Policy",
        "default-src 'none'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "style-src 'self' https://cdn.jsdelivr.net; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "font-src https://cdn.jsdelivr.net"
    )

def _append_message(user_text: str):
    global NEXT_ID, MESSAGES

    text = user_text.rstrip("\n")
    if text.strip() == ":quit":
        os._exit(0)

    if text.strip() == ":clear":
        global bot
        with LOCK:
            MESSAGES = []
            NEXT_ID = 1
        bot.reset()
        return

    
    reply = bot.send(text)
    rendered = safe_markdown(reply)
    
    with LOCK:
        mid = NEXT_ID
        NEXT_ID += 1
        MESSAGES.append(
            {
                "id": mid,
                "user_raw": text,
                "assistant_html": rendered,
            }
        )

# --- Secure Chat HTTP GET/POST Handler -----------------------------------------------------------

class SecureChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not _require_token(self):
            return

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            page = INDEX_HTML.format(token=SESSION_TOKEN).encode("utf-8")
            self.send_response(200)
            _send_common_headers(self, "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)
            return

        if path == "/app.css":
            self.send_response(200)
            _send_common_headers(self, "text/css; charset=utf-8")
            self.end_headers()
            self.wfile.write(APP_CSS.encode("utf-8"))
            return

        if path == "/app.js":
            self.send_response(200)
            _send_common_headers(self, "application/javascript; charset=utf-8")
            self.end_headers()
            self.wfile.write(APP_JS.encode("utf-8"))
            return

        if path == "/api/messages":
            q = parse_qs(parsed.query)
            after_id = 0
            try:
                after_id = int(q.get("after_id", ["0"])[0])
            except Exception:
                after_id = 0

            with LOCK:
                msgs = [m for m in MESSAGES if m["id"] > after_id]
                payload = {"messages": msgs[-200:]}  # keep responses bounded

            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            _send_common_headers(self, "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        _send_common_headers(self, "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Not Found\n")

    def do_POST(self):
        if not _require_token(self):
            return

        parsed = urlparse(self.path)
        if parsed.path != "/api/submit":
            self.send_response(404)
            _send_common_headers(self, "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Not Found\n")
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b""
        text = ""

        ctype = (self.headers.get("Content-Type") or "").lower()
        try:
            if "application/json" in ctype:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
                text = str(obj.get("text", ""))
            else:
                text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("utf-8", errors="replace")

        _append_message(text)

        self.send_response(200)
        _send_common_headers(self, "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(b'{"ok":true}\n')

    def log_message(self, *args):
        return  # 🔇 silence


# --- stdin reader in terminal (paste-friendly) ------------------------------------------



def stdin_reader():
    print("\n🔐 SECURE MARKDOWN CHAT")
    print("🌐 URL (open in your local browser):")
    print(f"   http://{HOST}:{PORT}/?token={SESSION_TOKEN}\n")
    print("📥 Paste text into this terminal; each paste becomes one chat turn.")
    print("   Commands: :clear  :quit\n")

    paste_buf = []
    last_input_time = None
    PASTE_GAP = 0.15  # seconds; tweak if needed

    while True:
        r, _, _ = select.select([sys.stdin], [], [], PASTE_GAP)

        if r:
            line = sys.stdin.readline()
            if not line:
                return
            line = line.rstrip("\n")
            paste_buf.append(line)
            last_input_time = time.time()
        else:
            if paste_buf:
                joined = "\n".join(paste_buf)
                _append_message(joined)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                paste_buf = []


# --- main ------------------------------------------------------------------

def main():
    threading.Thread(target=stdin_reader, daemon=True).start()
    server = HTTPServer((HOST, PORT), SecureChatHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
