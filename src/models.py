"""The model registry: one source of truth for ids, pinned revisions and loaders.

Revisions are pinned to commit hashes because every one of these models runs
custom code from its own repo under trust_remote_code=True. A pin means the
weights and that code cannot change underneath a half-finished run.
"""

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

MODELS = {
    "hyenadna-tiny": {
        "id": "LongSafari/hyenadna-tiny-1k-seqlen-hf",
        "revision": "e8c1effa8673814e257e627d2e1eda9ea5a373f6",
        "loader": AutoModelForCausalLM,
        "params": 438_144,
        # Character-level tokenizer: one token per base, so the window is
        # ~1 kb and 36% of E. coli protein-coding genes do not fit.
        "max_tokens": 1024,
    },
    "hyenadna-tiny-d256": {
        "id": "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
        "revision": "6036d4e144922b44470294f510cbfd2539ba3b7e",
        "loader": AutoModelForCausalLM,
        "params": 1_645_184,
        "max_tokens": 1024,
    },
    "nt-v2-50m": {
        "id": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "revision": "81b29e5786726d891dbf929404ef20adca5b36f1",
        "loader": AutoModelForMaskedLM,
        "params": 55_904_972,
        # 6-mer tokenizer, 2048 positions: about 12 kb, so every gene fits.
        "max_tokens": 2048,
    },
    "nt-v2-100m": {
        "id": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "revision": "f34324c6fde36a4f635f0f1f06cac5d25acd6798",
        "loader": AutoModelForMaskedLM,
        "params": 97_889_132,
        "max_tokens": 2048,
    },
}
