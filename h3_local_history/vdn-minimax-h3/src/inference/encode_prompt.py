"""Encode t2va prompts with the Qwen3-VL conditioner and cache the results.

Replicates `MiniMaxH3TextEncoderStep` exactly: the prompt verbatim, no chat template, no
special tokens, hidden state after decoder layer 50, tags all text. Run once on one GPU;
inference then reuses the cached file and never loads the 62 GB conditioner again.

    python src/inference/encode_prompt.py --prompt "..." --out prompts/my_prompt.pt
    python src/inference/encode_prompt.py --jsonl prompts.jsonl --take 4 \
        --out_dir prompts --name prompt

The batch form exists because loading the conditioner costs far more than encoding: four
separate --prompt runs would pay the 62 GB load four times. It also keeps a prompt set
reproducible -- the file it reads is the dataset as published, not text pasted onto a
command line.
"""

import argparse
import json
import os

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from src.paths import upstream_snapshot
TEXT_ENCODER_LAYER = 50
TEXT_TAG = 1


def encode(processor, text_encoder, prompt, out, device):
    token_ids = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    if hasattr(processor, "create_mm_token_type_ids"):
        mm_token_type_ids = torch.tensor(
            processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
        )
    else:
        # This is a text-only H3 prompt. Older Transformers processors do not expose
        # the helper; zero modality ids are the text-only contract used by the model.
        mm_token_type_ids = torch.zeros_like(input_ids, device=device)
    with torch.no_grad():
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    prompt_embeds = outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.bfloat16).cpu()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({
        "prompt": prompt,
        "prompt_embeds": prompt_embeds,  # (L, 5120) bf16
        "text_token_tags": torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long),
    }, out)
    print(f"wrote {out}: {len(token_ids)} tokens, embeds {tuple(prompt_embeds.shape)}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--jsonl", type=str,
                   help="one JSON object per line with a 'prompt' field; batch mode")
    p.add_argument("--take", type=int, default=4, help="jsonl: encode the first N rows")
    p.add_argument("--rows", type=int, nargs="*", default=None,
                   help="jsonl: encode exactly these row indices instead of the first "
                        "--take; output names keep the ORIGINAL index "
                        "(<name>_<row>.pt), so a prompt's identity is stable no matter "
                        "which subset a batch encodes")
    p.add_argument("--out_dir", type=str, default="prompts")
    p.add_argument("--name", type=str, default="prompt",
                   help="jsonl: outputs are <out_dir>/<name>_<i>.pt")
    p.add_argument("--model_root", type=str, default=None,
                   help="a MiniMax-H3 snapshot with processor/ and text_encoder/; "
                        "default: downloaded from the Hub on first use")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    if not args.jsonl and not (args.prompt and args.out):
        p.error("pass --jsonl, or both --prompt and --out")

    if args.model_root is None:
        args.model_root = upstream_snapshot("processor", "text_encoder")

    processor = Qwen3VLProcessor.from_pretrained(args.model_root, subfolder="processor")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_root, subfolder="text_encoder", dtype=torch.bfloat16
    ).to(args.device)
    text_encoder.eval().requires_grad_(False)

    if args.jsonl:
        with open(args.jsonl) as f:
            all_rows = [json.loads(l) for l in f]
        picked = (list(enumerate(all_rows))[:args.take] if args.rows is None
                  else [(i, all_rows[i]) for i in args.rows])
        for i, r in picked:
            encode(processor, text_encoder, r["prompt"],
                   os.path.join(args.out_dir, f"{args.name}_{i}.pt"), args.device)
    else:
        encode(processor, text_encoder, args.prompt, args.out, args.device)


if __name__ == "__main__":
    main()
