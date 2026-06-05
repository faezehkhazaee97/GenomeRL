"""DNABERT-2 tokenizer wrapper used as the BPE baseline."""

from __future__ import annotations

from typing import List

from transformers import AutoTokenizer


class DNABERT2BPETokenizer:
    """Thin wrapper around the DNABERT-2 Hugging Face tokenizer."""

    def __init__(self, model_name: str = "zhihan1996/DNABERT-2-117M"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def tokenize(self, sequence: str) -> List[str]:
        sequence = sequence.upper().strip()
        return self.tokenizer.tokenize(sequence)

    def encode(self, sequence: str):
        sequence = sequence.upper().strip()
        return self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )


def main() -> None:
    tokenizer = DNABERT2BPETokenizer()
    sequence = "ACGTACGTACGTACGT"
    tokens = tokenizer.tokenize(sequence)
    print("Sequence:", sequence)
    print("Tokens:", tokens)
    print("Number of tokens:", len(tokens))
    print("Compression ratio:", len(tokens) / len(sequence))


if __name__ == "__main__":
    main()
