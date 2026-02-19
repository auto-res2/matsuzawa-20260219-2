"""
Inference script for LtM prompt tuning experiments.
Implements LtM, LCV-LtM, and DiVA-LtM methods.
"""
import os
import sys
import json
import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.model import (
    LtMModel,
    build_decomposition_prompt,
    build_step_prompt,
    build_verification_prompt,
    build_final_prompt
)
from src.preprocess import (
    load_gsm8k,
    extract_final_answer,
    canonicalize_answer
)


def decompose_question(model: LtMModel, question: str, max_tokens: int, max_steps: int) -> List[str]:
    """Decompose question into subquestions."""
    prompt = build_decomposition_prompt(question)
    output = model.generate(prompt, temperature=0.0, max_new_tokens=max_tokens, do_sample=False)
    
    # Parse numbered subquestions
    subquestions = []
    lines = output.split('\n')
    for line in lines:
        # Match patterns like "1. ...", "1) ...", or "1: ..."
        match = re.match(r'^(\d+)[.):]\s*(.+)$', line.strip())
        if match and len(subquestions) < max_steps:
            subquestions.append(match.group(2).strip())
    
    # Ensure we have at least one subquestion
    if not subquestions:
        subquestions = [question]
    
    return subquestions


def run_ltm(
    model: LtMModel,
    example: Dict[str, Any],
    cfg: DictConfig
) -> Dict[str, Any]:
    """Run standard LtM (single sample per step)."""
    question = example["question"]
    
    # Decompose
    subquestions = decompose_question(
        model, question,
        cfg.inference.max_decomp_tokens,
        cfg.method.max_steps
    )
    
    # Solve steps sequentially
    subanswers = []
    for i, subq in enumerate(subquestions):
        prompt = build_step_prompt(question, subquestions, i, subanswers)
        answer = model.generate(
            prompt,
            temperature=cfg.method.step_temperature,
            max_new_tokens=cfg.inference.max_step_tokens
        )
        subanswers.append(answer)
    
    # Final answer
    final_prompt = build_final_prompt(question, subquestions, subanswers)
    final_output = model.generate(
        final_prompt,
        temperature=0.0,
        max_new_tokens=cfg.inference.max_final_tokens,
        do_sample=False
    )
    
    predicted = extract_final_answer(final_output)
    
    return {
        "question": question,
        "subquestions": subquestions,
        "subanswers": subanswers,
        "final_output": final_output,
        "predicted": predicted,
        "gold": example["answer"]
    }


def run_lcv_ltm(
    model: LtMModel,
    example: Dict[str, Any],
    cfg: DictConfig
) -> Dict[str, Any]:
    """Run LCV-LtM (fixed k samples with majority voting per step)."""
    question = example["question"]
    k = cfg.method.k_samples
    
    # Decompose
    subquestions = decompose_question(
        model, question,
        cfg.inference.max_decomp_tokens,
        cfg.method.max_steps
    )
    
    # Solve steps sequentially with voting
    subanswers = []
    all_candidates = []
    
    for i, subq in enumerate(subquestions):
        # Generate k candidates
        candidates = []
        for _ in range(k):
            prompt = build_step_prompt(question, subquestions, i, subanswers)
            answer = model.generate(
                prompt,
                temperature=cfg.method.step_temperature,
                max_new_tokens=cfg.inference.max_step_tokens
            )
            candidates.append(answer)
        
        # Majority vote with canonicalization
        canonical_candidates = [canonicalize_answer(c) for c in candidates]
        counter = Counter(canonical_candidates)
        most_common_canonical = counter.most_common(1)[0][0]
        
        # Find original answer matching most common canonical
        chosen_answer = candidates[0]  # default
        for orig, canon in zip(candidates, canonical_candidates):
            if canon == most_common_canonical:
                chosen_answer = orig
                break
        
        subanswers.append(chosen_answer)
        all_candidates.append(candidates)
    
    # Final answer
    final_prompt = build_final_prompt(question, subquestions, subanswers)
    final_output = model.generate(
        final_prompt,
        temperature=0.0,
        max_new_tokens=cfg.inference.max_final_tokens,
        do_sample=False
    )
    
    predicted = extract_final_answer(final_output)
    
    return {
        "question": question,
        "subquestions": subquestions,
        "subanswers": subanswers,
        "all_candidates": all_candidates,
        "final_output": final_output,
        "predicted": predicted,
        "gold": example["answer"]
    }


def compute_disagreement(canonical_answers: List[str]) -> float:
    """Compute disagreement (entropy) among answers."""
    if not canonical_answers:
        return 0.0
    
    counter = Counter(canonical_answers)
    total = len(canonical_answers)
    entropy = 0.0
    
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by max entropy
    max_entropy = math.log2(min(total, len(counter)))
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def run_diva_ltm(
    model: LtMModel,
    example: Dict[str, Any],
    cfg: DictConfig
) -> Dict[str, Any]:
    """Run DiVA-LtM (diverse-verified adaptive)."""
    question = example["question"]
    k_min = cfg.method.k_min
    k_max = cfg.method.k_max
    operators = cfg.method.cognitive_operators
    disagreement_threshold = cfg.method.disagreement_threshold
    
    # Decompose
    subquestions = decompose_question(
        model, question,
        cfg.inference.max_decomp_tokens,
        cfg.method.max_steps
    )
    
    # Solve steps sequentially with adaptive k
    subanswers = []
    all_candidates = []
    all_verified = []
    k_used = []
    
    for i, subq in enumerate(subquestions):
        k_current = k_min
        verified_answers = []
        verified_canonical = []
        candidates = []  # initialize
        previous_context = "\n".join([
            f"Subquestion {j+1}: {subquestions[j]}\nAnswer {j+1}: {subanswers[j]}"
            for j in range(i)
        ])
        
        while k_current <= k_max:
            # Generate candidates with operator diversity
            candidates = []
            num_per_operator = max(1, k_current // len(operators))
            
            for op in operators:
                for _ in range(num_per_operator):
                    if len(candidates) >= k_current:
                        break
                    prompt = build_step_prompt(question, subquestions, i, subanswers, operator=op)
                    answer = model.generate(
                        prompt,
                        temperature=cfg.method.step_temperature,
                        max_new_tokens=cfg.inference.max_step_tokens
                    )
                    candidates.append((answer, op))
                if len(candidates) >= k_current:
                    break
            
            # Verify each candidate
            for answer, op in candidates:
                verify_prompt = build_verification_prompt(
                    question, subq, answer, previous_context
                )
                verify_output = model.generate(
                    verify_prompt,
                    temperature=cfg.method.verification_temperature,
                    max_new_tokens=cfg.method.max_verify_tokens,
                    do_sample=False
                )
                
                # Parse verification
                if "VALID" in verify_output.upper() and "INVALID" not in verify_output.upper():
                    # Extract normalized answer
                    norm_match = re.search(r"Normalized answer:\s*(.+?)(?:\n|$)", verify_output, re.IGNORECASE)
                    if norm_match:
                        normalized = norm_match.group(1).strip()
                    else:
                        normalized = answer
                    
                    verified_answers.append(answer)
                    verified_canonical.append(canonicalize_answer(normalized))
            
            # Check disagreement
            if verified_canonical:
                disagreement = compute_disagreement(verified_canonical)
                if disagreement < disagreement_threshold or k_current >= k_max:
                    break
            
            # Increase k
            k_current = min(k_current + 2, k_max)
        
        # Vote among verified answers
        chosen_answer = "Unknown"  # default
        if verified_canonical:
            counter = Counter(verified_canonical)
            most_common_canonical = counter.most_common(1)[0][0]
            
            # Find original answer
            for orig, canon in zip(verified_answers, verified_canonical):
                if canon == most_common_canonical:
                    chosen_answer = orig
                    break
        elif candidates:
            # Fallback: use first candidate if no valid answers
            chosen_answer = candidates[0][0]
        
        subanswers.append(chosen_answer)
        all_candidates.append([c[0] for c in candidates])
        all_verified.append(verified_answers)
        k_used.append(k_current)
    
    # Final answer
    final_prompt = build_final_prompt(question, subquestions, subanswers)
    final_output = model.generate(
        final_prompt,
        temperature=0.0,
        max_new_tokens=cfg.inference.max_final_tokens,
        do_sample=False
    )
    
    predicted = extract_final_answer(final_output)
    
    return {
        "question": question,
        "subquestions": subquestions,
        "subanswers": subanswers,
        "all_candidates": all_candidates,
        "all_verified": all_verified,
        "k_used": k_used,
        "final_output": final_output,
        "predicted": predicted,
        "gold": example["answer"]
    }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main inference script."""
    
    # Apply mode overrides for sanity_check
    if cfg.mode == "sanity_check":
        # Reduce dataset size
        cfg.dataset.n_samples = 10
        # Keep wandb online but use separate project
        if not OmegaConf.is_missing(cfg.wandb, "project"):
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
    
    # Initialize wandb
    wandb_enabled = cfg.wandb.mode == "online"
    if wandb_enabled:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        print(f"WandB run: {wandb.run.url}")
    
    # Load model
    model = LtMModel(
        model_name=cfg.model.name,
        device=cfg.model.device,
        max_length=cfg.model.max_length,
        cache_dir=cfg.model.cache_dir
    )
    
    # Load dataset
    examples = load_gsm8k(
        split=cfg.dataset.split,
        n_samples=cfg.dataset.n_samples,
        seed=cfg.dataset.seed,
        cache_dir=cfg.model.cache_dir
    )
    
    print(f"Running {cfg.run.run_id} on {len(examples)} examples")
    
    # Run inference
    results = []
    correct = 0
    total = 0
    
    method_type = cfg.method.type
    
    for idx, example in enumerate(examples):
        print(f"Processing example {idx+1}/{len(examples)}")
        
        # Run appropriate method
        if method_type == "ltm":
            result = run_ltm(model, example, cfg)
        elif method_type == "lcv-ltm":
            result = run_lcv_ltm(model, example, cfg)
        elif method_type == "diva-ltm":
            result = run_diva_ltm(model, example, cfg)
        else:
            raise ValueError(f"Unknown method type: {method_type}")
        
        results.append(result)
        
        # Track accuracy
        if result["predicted"] is not None and result["gold"] is not None:
            if abs(result["predicted"] - result["gold"]) < 1e-5:
                correct += 1
            total += 1
        
        # Log to wandb
        if wandb_enabled:
            wandb.log({
                "example_idx": idx,
                "correct": result["predicted"] == result["gold"] if result["predicted"] is not None else False,
                "accuracy": correct / total if total > 0 else 0.0
            })
    
    # Final metrics
    accuracy = correct / total if total > 0 else 0.0
    print(f"Final accuracy: {accuracy:.4f} ({correct}/{total})")
    
    if wandb_enabled:
        wandb.summary["accuracy"] = accuracy
        wandb.summary["correct"] = correct
        wandb.summary["total"] = total
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }, f, indent=2)
    
    # Sanity validation for sanity_check mode
    if cfg.mode == "sanity_check":
        # Check that at least 5 samples were processed
        passed = True
        reasons = []
        
        if total < 5:
            passed = False
            reasons.append("insufficient_samples")
        
        # Check that accuracy is not always 0
        if total > 0 and correct == 0 and total >= 5:
            passed = False
            reasons.append("zero_accuracy")
        
        # Check for finite metrics
        if not math.isfinite(accuracy):
            passed = False
            reasons.append("non_finite_metrics")
        
        # Print validation result
        if passed:
            print("SANITY_VALIDATION: PASS")
        else:
            print(f"SANITY_VALIDATION: FAIL reason={','.join(reasons)}")
        
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'correct': correct, 'accuracy': accuracy})}")
    
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
