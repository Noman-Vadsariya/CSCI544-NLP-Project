# src/tokenizer_pipeline.py

import torch
from transformers import AutoTokenizer


class TokenizerPipeline:
    """
    Tokenization + chunking pipeline for Doc-to-LoRA training.
    Handles:
        - context tokenization
        - query tokenization
        - response tokenization
        - long context chunking
    """

    def __init__(self, model_name="distilgpt2"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # GPT-style models often don't define pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ----------------------------------------------------
    # Basic tokenization
    # ----------------------------------------------------

    def tokenize_context(self, context, max_length=1024):

        return self.tokenizer(
            context,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def tokenize_query(self, query, max_length=128):

        return self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def tokenize_response(self, response, max_length=128):

        return self.tokenizer(
            response,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    # ----------------------------------------------------
    # Long context chunking (Doc-to-LoRA requirement)
    # ----------------------------------------------------

    def chunk_context(self, context, chunk_size=512, stride=0):
        """
        Convert context string into token chunks.

        Parameters
        ----------
        context : str
        chunk_size : int
        stride : int
            overlap between chunks

        Returns
        -------
        chunks : list[Tensor]
        attention_masks : list[Tensor]
        """

        encoding = self.tokenizer(
            context,
            truncation=False,
            return_attention_mask=False
        )

        token_ids = encoding["input_ids"]

        # ensure Python list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chunks = []
        masks = []

        i = 0
        while i < len(token_ids):

            chunk_ids = token_ids[i:i + chunk_size]

            # Pad if final chunk smaller
            padded = self.tokenizer.pad(
                {"input_ids": [chunk_ids]},
                padding="max_length",
                max_length=chunk_size,
                return_tensors="pt"
            )

            chunks.append(padded["input_ids"][0])
            masks.append(padded["attention_mask"][0])

            if stride > 0:
                i += chunk_size - stride
            else:
                i += chunk_size

        return chunks, masks

    # ----------------------------------------------------
    # Utility debugging function
    # ----------------------------------------------------

    def debug_tokenization(self, text):

        enc = self.tokenizer(text)

        print("\nOriginal text:")
        print(text[:100])

        print("\nToken IDs:")
        print(enc["input_ids"][:20])

        print("\nDecoded back:")
        print(self.tokenizer.decode(enc["input_ids"][:20]))