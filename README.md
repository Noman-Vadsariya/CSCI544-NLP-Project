# CSCI544-NLP-Project

## Environment Setup

### Prerequisites
- Python 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA 12.4 compatible GPU 

### Installation

2. Create and activate a virtual environment:
```bash
uv venv --python 3.10
source .venv/bin/activate
```

3. Install PyTorch with CUDA 12.4 support:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4. Install the remaining dependencies:
```bash
uv pip install -r requirements.txt
```

> **Note:** `flash-attn` in `requirements.txt` is optional. Check if you support it


### Testing
- Run following commads in root directory to test
    - pip install -e . 
    - python tests/test_dataset_loading.py
    - python tests/test_tokenizer_chunking.py
