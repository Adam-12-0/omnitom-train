#!/usr/bin/env python3
"""Held-out OmniToM Stage-1 generation and exact structured metrics."""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-14B"

def norm(s: str) -> str:
    return " ".join(s.lower().strip().strip("`| ").split())

def rows(text: str):
    result=[]
    for line in text.splitlines():
        parts=[p.strip() for p in line.split("|")]
        if len(parts)==3 and norm(parts[0]) not in {"actor", ""}:
            result.append(tuple(norm(x) for x in parts))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT/"data/omnitom_stage1_sft_stratified/validation.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if args.adapter: model=PeftModel.from_pretrained(model, str(args.adapter))
    model.eval(); outputs=[]; tp=fp=fn=actor_ok=actor_total=order_ok=order_total=exact=0
    for line in args.data.read_text().splitlines():
        ex=json.loads(line); prompt=ex["messages"][:-1]; gold=ex["messages"][-1]["content"]
        # `apply_chat_template` returns a BatchEncoding in this Transformers
        # version. Pass its tensor fields to generation rather than the
        # container itself (which has no `.shape`).
        encoded=tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt",
                                         enable_thinking=False).to(model.device)
        ids=encoded["input_ids"]
        with torch.inference_mode(): generated=model.generate(**encoded, max_new_tokens=min(args.max_new_tokens, 1024), do_sample=False, pad_token_id=tok.eos_token_id)
        pred=tok.decode(generated[0,ids.shape[-1]:], skip_special_tokens=True)
        g=set(rows(gold)); p=set(rows(pred)); tp+=len(p&g); fp+=len(p-g); fn+=len(g-p)
        exact+=int(p==g)
        # Attribute accuracy is measured conditionally on the other two fields:
        # actor given (belief, order), and order given (actor, belief).  This
        # prevents a proposition exact match from being reported as both scores.
        gold_by_belief_order = {(b, o): a for a, b, o in g}
        pred_by_belief_order = {(b, o): a for a, b, o in p}
        for key, actor in pred_by_belief_order.items():
            if key in gold_by_belief_order:
                actor_total += 1
                actor_ok += int(actor == gold_by_belief_order[key])
        gold_by_actor_belief = {(a, b): o for a, b, o in g}
        pred_by_actor_belief = {(a, b): o for a, b, o in p}
        for key, order in pred_by_actor_belief.items():
            if key in gold_by_actor_belief:
                order_total += 1
                order_ok += int(order == gold_by_actor_belief[key])
        outputs.append({"id":ex["id"],"prediction":pred,"gold":gold})
        if len(outputs) % 5 == 0:
            print(f"processed {len(outputs)}/{len(args.data.read_text().splitlines())}", flush=True)
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
    metrics={"n":len(outputs),"exact_match":exact/len(outputs),"proposition_precision":precision,"proposition_recall":recall,"proposition_f1":2*precision*recall/(precision+recall) if precision+recall else 0,"actor_accuracy":actor_ok/actor_total if actor_total else 0,"actor_comparable":actor_total,"order_accuracy":order_ok/order_total if order_total else 0,"order_comparable":order_total,"tp":tp,"fp":fp,"fn":fn}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps({"metrics":metrics,"outputs":outputs},indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
