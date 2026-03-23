import os
import re
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================

REPO_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2.5-Coder-14B-Instruct"

MEM_CONTEXT_TURNS = 20  # each turn = (user + assistant)

# Enforce offline behavior
os.environ["HF_HUB_OFFLINE"] = "1"

# =========================
# DOWNLOAD (ONE-TIME)
# =========================

if not os.path.exists(LOCAL_MODEL_DIR):
    print(f"[INFO] Downloading model snapshot to {LOCAL_MODEL_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    print("[INFO] Download complete.")
else:
    print("[INFO] Using existing local model directory.")

# =========================
# LOAD TOKENIZER + MODEL
# =========================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

print("Loading model (4-bit quantized, offline)...")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True,
)

model.eval()

# =========================
# CHAT STATE
# =========================

conversation = []

SYSTEM_PROMPT = (
    "You are an AI assistant strong in math, coding, and AI research. "
    "You answer carefully and precisely.\n\n"

    "Formatting rules (MANDATORY):\n\n"

    "- Use Markdown for all answers."
    "- Inline math MUST use \( ... \)."
    "- Display math MUST use $$ ... $$."
    "- NEVER use $ ... $ for inline math."
    "- NEVER use \[ ... \]."
    "- Do not split LaTeX environments across text blocks."

    "IF you write any code, you carefully & thoroughly ensure it is correct, clean, and well-structured."
)

# =========================
# FORMATTING UTILITIES
# =========================

def normalize_latex(text: str) -> str:
    """
    Normalize LaTeX:
    - \\[ ... \\]  ->  $$ ... $$
    - \\( ... \\)  ->  $ ... $
    """
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "$$\n" + m.group(1).strip() + "\n$$",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"\\\((.*?)\\\)",
        r"$\1$",
        text,
        flags=re.DOTALL,
    )

    return text


def bold_short_inline_math(text: str, max_len: int = 500) -> str:
    """
    Bold inline math ($...$) if short.
    Does NOT affect $$...$$ blocks.
    """

    def replacer(match):
        content = match.group(1)
        if len(content) < max_len:
            return f"**${content}$**"
        return f"${content}$"

    return re.sub(
        r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
        replacer,
        text,
        flags=re.DOTALL,
    )


def format_assistant_output(text: str) -> str:
    """
    Enforce math contract:
    - Inline math: \( ... \)
    - Display math: $$ ... $$
    """

    # Normalize any legacy math to our standard
    text = re.sub(
        r"\$(?!\$)(.*?)\$(?!\$)",
        r"\\(\1\\)",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"\\\[(.*?)\\\]",
        r"$$\n\1\n$$",
        text,
        flags=re.DOTALL,
    )

    return text



# =========================
# PROMPT BUILDING
# =========================

def build_prompt(user_input, conversation):
    recent_turns = conversation[-MEM_CONTEXT_TURNS:]
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n"
    for u, a in recent_turns:
        prompt += f"<|user|>\n{u}\n<|assistant|>\n{a}\n"
    prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"
    return prompt


@torch.no_grad()
def generate_reply(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=5120,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded[len(prompt):].strip()
    return reply


# =========================
# INTERACTIVE LOOP
# =========================

class ChatBot:
    def __init__(self):
        self.conversation = []


    def send(self, user_input: str) -> str:
        prompt = build_prompt(user_input, self.conversation)
        reply = generate_reply(prompt)
        reply = format_assistant_output(reply)
        self.conversation.append((user_input, reply))
        return reply
    
    def reset(self):
        self.conversation.clear()


def main():
    global conversation
    print("\nQwen2.5-Coder-14B (OFFLINE) Interactive Terminal Chat")
    print("Type 'exit' or 'quit' to stop.\n")
    bot = ChatBot()
    while True:
        user_input = input().strip()
        if user_input.lower() in {"exit", "quit", ":exit", ":quit"}:
            break
        if user_input.lower() in {"clear", ":clear"}:
            conversation = []
            print("[INFO] Conversation history cleared.\n")
            continue
        
        reply = bot.send(user_input)

        print("\n" + reply + "\n")


if __name__ == "__main__":
    print()
    main()
    print()


