"""Turn a gene sequence into a fixed-length vector, and never compute one twice.

Pooling
    mean   average over all token positions
    last   the final token's hidden state

Sequences longer than the model's window
    HyenaDNA's window is about 1 kb, and 36% of E. coli protein-coding genes
    are longer, so this choice is not a detail. It is a parameter:

      chunk_mean  split into consecutive windows, embed each, average them
                  weighted by token count. Uses the whole gene. The default.
      truncate    keep the first window only. What most papers do silently.
      center      keep the middle window.

    Nucleotide Transformer's 6-mer tokenizer reaches about 12 kb, so no gene
    hits its limit and the three strategies agree for it.

Caching
    One .npz per (model, pooling, strategy), keyed by locus tag, written
    incrementally. A crashed or interrupted run resumes instead of starting
    over.
"""

import inspect
import time
from pathlib import Path

import numpy as np
import torch

from .models import MODELS

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
POOLINGS = ("mean", "last")
STRATEGIES = ("chunk_mean", "truncate", "center")


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cache_path(model_key, pooling, strategy):
    return CACHE_DIR / f"{model_key}__{pooling}__{strategy}.npz"


class EmbeddingCache:
    """A dict of locus_tag -> vector, persisted as a single .npz."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.data = {k: z[k] for k in z.files}

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def add(self, key, vector):
        self.data[key] = np.asarray(vector, dtype=np.float32)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        # Write through a handle: np.savez_compressed appends ".npz" to a path
        # that lacks it, which would leave the temp file somewhere else.
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **self.data)
        tmp.replace(self.path)          # atomic, so a crash cannot truncate it


class Embedder:
    def __init__(self, model_key, device=None):
        if model_key not in MODELS:
            raise SystemExit(f"unknown model {model_key!r}; choose from {sorted(MODELS)}")
        from transformers import AutoTokenizer

        spec = MODELS[model_key]
        self.key = model_key
        self.spec = spec
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["id"], revision=spec["revision"], trust_remote_code=True)
        self.model = spec["loader"].from_pretrained(
            spec["id"], revision=spec["revision"], trust_remote_code=True).to(self.device).eval()
        self.accepted = inspect.signature(self.model.forward).parameters
        self.max_tokens = spec["max_tokens"]
        print(f"[{model_key}] loaded on {self.device}, "
              f"{sum(p.numel() for p in self.model.parameters()):,} params, "
              f"window {self.max_tokens} tokens")

    def _token_ids(self, seq):
        """Tokenise without special tokens so windows can be cut cleanly."""
        return self.tokenizer(seq, add_special_tokens=False)["input_ids"]

    def _forward(self, ids):
        batch = {"input_ids": torch.tensor([ids], device=self.device)}
        if "attention_mask" in self.accepted:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        batch = {k: v for k, v in batch.items() if k in self.accepted}
        with torch.inference_mode():
            out = self.model(**batch, output_hidden_states=True)
        return out.hidden_states[-1][0]          # (tokens, hidden)

    def _windows(self, ids, strategy):
        limit = self.max_tokens
        if len(ids) <= limit:
            return [ids]
        if strategy == "truncate":
            return [ids[:limit]]
        if strategy == "center":
            start = (len(ids) - limit) // 2
            return [ids[start:start + limit]]
        if strategy == "chunk_mean":
            chunks = [ids[i:i + limit] for i in range(0, len(ids), limit)]
            # A final sliver would be dominated by edge effects; fold it into
            # the previous window by taking the last full window instead.
            if len(chunks) > 1 and len(chunks[-1]) < limit // 4:
                chunks[-1] = ids[-limit:]
            return chunks
        raise ValueError(f"unknown strategy {strategy!r}")

    def embed(self, seq, pooling="mean", strategy="chunk_mean"):
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}")
        ids = self._token_ids(seq.upper())
        if not ids:
            raise ValueError("empty sequence")
        windows = self._windows(ids, strategy)
        vectors, weights = [], []
        for window in windows:
            hidden = self._forward(window)
            vectors.append((hidden.mean(dim=0) if pooling == "mean" else hidden[-1]).float().cpu().numpy())
            weights.append(len(window))
        if len(vectors) == 1:
            return vectors[0].astype(np.float32)
        if pooling == "last":
            return vectors[-1].astype(np.float32)   # the gene's true final token
        w = np.asarray(weights, dtype=np.float64)
        return np.average(np.stack(vectors), axis=0, weights=w).astype(np.float32)

    def embed_many(self, items, pooling="mean", strategy="chunk_mean",
                   cache=None, checkpoint_every=200, log_every=200):
        """items: iterable of (locus_tag, sequence). Returns the cache."""
        cache = cache if cache is not None else EmbeddingCache(
            cache_path(self.key, pooling, strategy))
        todo = [(t, s) for t, s in items if t not in cache]
        print(f"[{self.key}] {len(cache)} cached, {len(todo)} to compute")
        start = time.perf_counter()
        for i, (tag, seq) in enumerate(todo, 1):
            cache.add(tag, self.embed(seq, pooling, strategy))
            if i % checkpoint_every == 0:
                cache.save()
            if i % log_every == 0 or i == len(todo):
                rate = i / (time.perf_counter() - start)
                left = (len(todo) - i) / rate if rate else 0
                print(f"   {i}/{len(todo)}  {rate:.1f} genes/s  ~{left / 60:.1f} min left", flush=True)
        cache.save()
        return cache
