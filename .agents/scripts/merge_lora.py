"""Merge LoRA adapters into full checkpoints so the canonical scorer can load them.

The canonical LLMLingua2FixedThreshold loads via AutoModelForTokenClassification
.from_pretrained, which cannot read a bare PEFT adapter dir. merge_and_unload folds the
adapter into base weights; the merged dir then loads exactly like the stock model.

Usage (GPU box): ./venv/bin/python merge_lora.py --corpus swe-bench
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("merge_lora")
HERE = Path(__file__).resolve().parent
MODEL_ID = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    adapter = HERE / f"adapted-{args.corpus}-lora"
    out = HERE / f"adapted-{args.corpus}-lora-merged"
    base = AutoModelForTokenClassification.from_pretrained(MODEL_ID, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(str(out))
    log.info("merged %s -> %s", adapter, out)


if __name__ == "__main__":
    main()
