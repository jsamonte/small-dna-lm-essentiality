"""Prove each model in CLAUDE.md loads on this machine and produces an embedding.

For every model: load it, count its parameters, run one forward pass on a
seeded random 500-base sequence, mean-pool the last hidden layer into an
embedding, and time the forward pass. A model that fails is reported with its
error, not skipped silently.

Writes results/model_check.csv and prints the same table.

    .venv\\Scripts\\python scripts\\check_models.py [--device cpu] [--repeats 5]
"""

import argparse
import csv
import inspect
import os
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

SEED = 0
SEQ_LEN = 500

# Loader class per model card: HyenaDNA is causal, Nucleotide Transformer is masked.
# Revisions are pinned because trust_remote_code runs the repo's Python: a pinned
# commit means the weights and code cannot change under us.
MODELS = [
    ("LongSafari/hyenadna-tiny-1k-seqlen-hf", "e8c1effa8673814e257e627d2e1eda9ea5a373f6", AutoModelForCausalLM),
    ("LongSafari/hyenadna-tiny-1k-seqlen-d256-hf", "6036d4e144922b44470294f510cbfd2539ba3b7e", AutoModelForCausalLM),
    ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", "81b29e5786726d891dbf929404ef20adca5b36f1", AutoModelForMaskedLM),
    ("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "f34324c6fde36a4f635f0f1f06cac5d25acd6798", AutoModelForMaskedLM),
]

RESULTS = Path(__file__).resolve().parent.parent / "results" / "model_check.csv"


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or "unknown"


def ram_gb():
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "avail", "totalPage", "availPage", "totalVirtual", "availVirtual", "availExtended")
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total / 2**30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def report_machine(device):
    print(f"OS:            {platform.platform()}")
    print(f"CPU:           {cpu_name()} ({os.cpu_count()} logical cores)")
    print(f"RAM:           {ram_gb():.1f} GB")
    print(f"Python:        {platform.python_version()} ({platform.machine()})")
    print(f"torch:         {torch.__version__}, {torch.get_num_threads()} threads")
    print(f"transformers:  {transformers.__version__}")
    print(f"CUDA:          {torch.cuda.is_available()}")
    print(f"MPS:           {torch.backends.mps.is_available()}")
    print(f"Device used:   {device}")
    print()


def dummy_sequence():
    rng = random.Random(SEED)
    return "".join(rng.choice("ACGT") for _ in range(SEQ_LEN))


def check(model_id, revision, loader, sequence, device, repeats):
    row = {"model": model_id, "params": "", "loaded": "no", "embed_dim": "", "tokens": "",
           "sec_per_seq": "", "deterministic": "", "revision": revision, "error": ""}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        model = loader.from_pretrained(model_id, revision=revision, trust_remote_code=True).to(device).eval()
        row["params"] = sum(p.numel() for p in model.parameters())
        row["loaded"] = "yes"

        # Pass only the inputs this model's forward() accepts; tokenizers differ.
        accepted = inspect.signature(model.forward).parameters
        encoded = tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items() if k in accepted}
        row["tokens"] = inputs["input_ids"].shape[1]

        def embed():
            with torch.inference_mode():
                out = model(**inputs, output_hidden_states=True)
            return out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu()

        first = embed()  # warm-up, also the reference for the determinism check
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            again = embed()
            times.append(time.perf_counter() - start)
        row["embed_dim"] = first.shape[0]
        row["sec_per_seq"] = round(statistics.median(times), 4)
        row["deterministic"] = "yes" if torch.equal(first, again) else "no"
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="default: cuda, then mps, then cpu")
    parser.add_argument("--repeats", type=int, default=5, help="timed forward passes per model (median reported)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = pick_device(args.device)
    report_machine(device)

    sequence = dummy_sequence()
    rows = []
    for model_id, revision, loader in MODELS:
        print(f"--- {model_id}", flush=True)
        rows.append(check(model_id, revision, loader, sequence, device, args.repeats))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("| model | params | loaded | embed dim | tokens | sec/seq | deterministic | error |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        params = f"{r['params']:,}" if r["params"] != "" else ""
        print(f"| {r['model']} | {params} | {r['loaded']} | {r['embed_dim']} | {r['tokens']} | "
              f"{r['sec_per_seq']} | {r['deterministic']} | {r['error']} |")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
