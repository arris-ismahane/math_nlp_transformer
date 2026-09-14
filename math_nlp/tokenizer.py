"""
Digit-level tokenizer: each digit and symbol gets its own token, instead of BPE, 
so place value stays aligned and consistent across examples.
"""

from __future__ import annotations

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
BASE_CHARS = list("0123456789") + list("x=+,;|:->ARc")


class Tokenizer:
    def __init__(self):
        vocab = SPECIAL_TOKENS + BASE_CHARS
        assert len(set(vocab)) == len(vocab), "duplicate symbol in vocab"
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.vocab_size = len(vocab)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"character {ch!r} not in tokenizer vocab: {text!r}")
            ids.append(self.stoi[ch])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, tokens, stop_at_eos: bool = True) -> str:
        chars = []
        for i in tokens:
            i = int(i)
            if stop_at_eos and i == self.eos_id:
                break
            if i == self.pad_id or i == self.bos_id:
                continue
            chars.append(self.itos[i])
        return "".join(chars)
