"""Thin CLI entrypoint cho Exercise 1: Tokenizer Anatomy & Hugging Face Sentiment Pipeline."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab3_v2.config as cfg
from src.vibe_scripts.lab3_v2.ex1_sentiment import DEFAULT_SAMPLE_SENTENCE, run_ex1_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 3 v2 Exercise 1 — WordPiece Tokenizer Anatomy & Sentiment Analysis Pipeline"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=cfg.EX1_MODEL_NAME,
        help=f"Model ID trên Hugging Face Hub (mặc định: {cfg.EX1_MODEL_NAME}).",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default=DEFAULT_SAMPLE_SENTENCE,
        help="Câu mẫu để giải phẫu tokenizer (tokens, input_ids, attention_mask, decode).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Ép chạy trên CPU thay vì GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_id = -1 if args.cpu else None
    result = run_ex1_demo(
        model_name=args.model_name,
        output_dir=cfg.EX1_DIR,
        sample_sentence=args.sentence,
        device=device_id,
    )

    tok = result["tokenizer_demo"]
    print("\n=== [EX1] TOKENIZER ANATOMY ===")
    print(f"Original   : {tok['original_text']}")
    print(f"Num tokens : {tok['num_tokens']} | Subwords (##): {tok['subword_pieces']}")
    print(f"Tokens     : {tok['tokens']}")
    print(f"input_ids  : {tok['input_ids']}")
    print(f"attn_mask  : {tok['attention_mask']}")
    print(f"Decode raw : {tok['decoded_with_special']}")
    print(f"Decode cln : {tok['decoded_clean']}")

    bp = result["batch_padding_demo"]
    print(f"\n=== [EX1] BATCH PADDING DEMO (shape={bp['input_ids_shape']}, dtype={bp['dtype']}) ===")
    for row in bp["rows"]:
        print(f"  real={row['real_tokens']:2d} | pad={row['pad_tokens']:2d} | text={row['text']}")

    print(f"\n=== [EX1] SENTIMENT PREDICTIONS ({result['matched_expected']}/{result['num_samples']} matched expected) ===")
    for pred in result["predictions"]:
        mark = "OK" if pred["matches_expected"] else "MISS"
        print(
            f"  [{mark:4s}] ({pred['category']:17s}) Pred={pred['predicted_label']:8s} "
            f"({pred['score']*100:5.2f}%) | Exp={pred['expected']} | {pred['text'][:72]}"
        )
    print(f"\nSaved JSON: {cfg.EX1_DIR / 'ex1_results.json'}")


if __name__ == "__main__":
    main()
