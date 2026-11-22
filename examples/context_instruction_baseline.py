"""Unfine-tuned instruction-model baseline for the feedback adaptation task."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.context_adaptation_benchmark import make_trial, evaluate


class InstructionPolicy:
    def __init__(self, model, revision, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.policy = (
            AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                torch_dtype=torch.float32,
                trust_remote_code=False,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.update_steps = 0
        tokens = [
            self.tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")
        ]
        if any(len(x) != 1 for x in tokens):
            raise ValueError("baseline requires single-token A/B answers")
        self.answer_tokens = [x[0] for x in tokens]

    @torch.no_grad()
    def action_results_structured(self, observations, *, deterministic=True):
        texts = []
        for obs in observations:
            context = obs.metadata["llm"]
            options = "\n".join(
                f'{letter}: {key}: {context["candidate_descriptions"][key]}'
                for letter, key in zip(("A", "B"), obs.candidate_ids)
            )
            past = (
                "\n".join(
                    f'{e["observation"]} Action: {e["action"]}. Result: {e["outcome"]}. Reward: {e["reward"]}.'
                    for e in context["history"]
                )
                or "No previous attempts."
            )
            prompt = f'{context["observation"]}\nCandidates:\n{options}\nPast attempts, oldest first:\n{past}\nSelect the next candidate. Answer with only A or B.'
            texts.append(
                self.tokenizer.apply_chat_template(
                    [dict(role="user", content=prompt)],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        results = []
        for start in range(0, len(texts), 8):
            tokens = self.tokenizer(
                texts[start : start + 8], return_tensors="pt", padding=True
            ).to(self.device)
            logits = (
                self.policy(**tokens).logits[:, -1, self.answer_tokens].float().cpu()
            )
            for obs, values in zip(observations[start : start + 8], logits):
                index = int(values.argmax())
                results.append(
                    SimpleNamespace(
                        candidate_id=obs.candidate_ids[index],
                        candidate_values=tuple(values.tolist()),
                    )
                )
        return tuple(results)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(json.dumps(vars(args), indent=2))
    torch.set_num_threads(2)
    agent = InstructionPolicy(args.model, args.revision, args.device)
    worlds = [
        replace(
            make_trial(args.seed + 100000, args.split, i, 12),
            group_history=args.group_history,
        )
        for i in range(24)
    ]
    result = {
        method: evaluate(agent, worlds, 1, out / f"{method}.jsonl", method=method)
        for method in ("history", "no_history", "flipped_feedback")
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=3109)
    p.add_argument("--split", type=int, default=2)
    p.add_argument("--group-history", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--revision", default="12fd25f77366fa6b3b4b768ec3050bf629380bac")
    run(p.parse_args())
