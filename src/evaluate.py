"""
Evaluation script for comparing LtM methods.
Fetches metrics from WandB and generates comparison plots.
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """Fetch run data from WandB API."""
    api = wandb.Api()
    
    # Find run by ID
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_id})
    
    if not runs:
        # Try by run ID directly
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except:
            raise ValueError(f"Run {run_id} not found in {entity}/{project}")
    else:
        run = runs[0]
    
    # Get config, summary, and history
    config = run.config
    summary = run.summary._json_dict
    
    # Get history (step-wise metrics)
    history = []
    for row in run.scan_history():
        history.append(row)
    
    return {
        "run_id": run_id,
        "config": config,
        "summary": summary,
        "history": history
    }


def export_per_run_metrics(results_dir: Path, run_data: Dict[str, Any]):
    """Export per-run metrics to JSON."""
    run_id = run_data["run_id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export summary metrics
    metrics = {
        "run_id": run_id,
        "accuracy": run_data["summary"].get("accuracy", 0.0),
        "correct": run_data["summary"].get("correct", 0),
        "total": run_data["summary"].get("total", 0)
    }
    
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics for {run_id}")
    
    # Create per-run figure (accuracy over examples)
    if run_data["history"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        examples = [row.get("example_idx", i) for i, row in enumerate(run_data["history"])]
        accuracies = [row.get("accuracy", 0.0) for row in run_data["history"]]
        
        ax.plot(examples, accuracies, marker='o', markersize=3, linewidth=1.5)
        ax.set_xlabel("Example Index")
        ax.set_ylabel("Cumulative Accuracy")
        ax.set_title(f"Accuracy Progress - {run_id}")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(run_dir / "accuracy_progress.pdf", format='pdf')
        plt.close(fig)
        
        print(f"Generated figure: {run_dir / 'accuracy_progress.pdf'}")


def create_comparison_plots(results_dir: Path, all_run_data: List[Dict[str, Any]]):
    """Create comparison plots across all runs."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Bar plot of final accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = [rd["run_id"] for rd in all_run_data]
    accuracies = [rd["summary"].get("accuracy", 0.0) for rd in all_run_data]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors[:len(run_ids)])
    
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Accuracy Comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(comparison_dir / "comparison_accuracy.pdf", format='pdf')
    plt.close(fig)
    
    print(f"Generated figure: {comparison_dir / 'comparison_accuracy.pdf'}")
    
    # Line plot of accuracy progress (if history available)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, rd in enumerate(all_run_data):
        if rd["history"]:
            examples = [row.get("example_idx", j) for j, row in enumerate(rd["history"])]
            accuracies = [row.get("accuracy", 0.0) for row in rd["history"]]
            
            ax.plot(examples, accuracies,
                   label=rd["run_id"],
                   color=colors[i % len(colors)],
                   linestyle=linestyles[i % len(linestyles)],
                   linewidth=2,
                   marker='o' if len(examples) < 50 else None,
                   markersize=4 if len(examples) < 50 else 0)
    
    ax.set_xlabel("Example Index")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Accuracy Progress Comparison")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(comparison_dir / "comparison_accuracy_progress.pdf", format='pdf')
    plt.close(fig)
    
    print(f"Generated figure: {comparison_dir / 'comparison_accuracy_progress.pdf'}")


def export_aggregated_metrics(results_dir: Path, all_run_data: List[Dict[str, Any]]):
    """Export aggregated comparison metrics."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize metrics by run_id
    metrics_by_run = {}
    for rd in all_run_data:
        run_id = rd["run_id"]
        metrics_by_run[run_id] = {
            "accuracy": rd["summary"].get("accuracy", 0.0),
            "correct": rd["summary"].get("correct", 0),
            "total": rd["summary"].get("total", 0)
        }
    
    # Identify proposed vs baselines
    proposed_runs = [rid for rid in metrics_by_run.keys() if "proposed" in rid]
    baseline_runs = [rid for rid in metrics_by_run.keys() if "comparative" in rid]
    
    best_proposed = None
    best_proposed_acc = 0.0
    if proposed_runs:
        for rid in proposed_runs:
            acc = metrics_by_run[rid]["accuracy"]
            if acc > best_proposed_acc:
                best_proposed = rid
                best_proposed_acc = acc
    
    best_baseline = None
    best_baseline_acc = 0.0
    if baseline_runs:
        for rid in baseline_runs:
            acc = metrics_by_run[rid]["accuracy"]
            if acc > best_baseline_acc:
                best_baseline = rid
                best_baseline_acc = acc
    
    gap = best_proposed_acc - best_baseline_acc if (best_proposed and best_baseline) else None
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics": metrics_by_run,
        "best_proposed": best_proposed,
        "best_proposed_accuracy": best_proposed_acc,
        "best_baseline": best_baseline,
        "best_baseline_accuracy": best_baseline_acc,
        "gap": gap
    }
    
    with open(comparison_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Exported aggregated metrics: {comparison_dir / 'aggregated_metrics.json'}")
    print(f"Best proposed: {best_proposed} ({best_proposed_acc:.4f})")
    print(f"Best baseline: {best_baseline} ({best_baseline_acc:.4f})")
    if gap is not None:
        print(f"Gap: {gap:.4f}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate and compare LtM runs")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True,
                       help="JSON string list of run IDs")
    parser.add_argument("--wandb_entity", type=str, default="airas",
                       help="WandB entity")
    parser.add_argument("--wandb_project", type=str, default="2026-02-19-2",
                       help="WandB project")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print(f"Evaluating runs: {run_ids}")
    
    # Fetch data for all runs
    all_run_data = []
    for run_id in run_ids:
        print(f"Fetching data for {run_id}...")
        try:
            run_data = fetch_run_data(args.wandb_entity, args.wandb_project, run_id)
            all_run_data.append(run_data)
        except Exception as e:
            print(f"Warning: Failed to fetch {run_id}: {e}")
            # Try to load from local metrics.json if available
            local_metrics = results_dir / run_id / "metrics.json"
            if local_metrics.exists():
                with open(local_metrics) as f:
                    metrics = json.load(f)
                all_run_data.append({
                    "run_id": run_id,
                    "config": {},
                    "summary": metrics,
                    "history": []
                })
            else:
                continue
    
    if not all_run_data:
        print("Error: No run data found")
        return 1
    
    # Export per-run metrics and figures
    for run_data in all_run_data:
        export_per_run_metrics(results_dir, run_data)
    
    # Create comparison plots
    create_comparison_plots(results_dir, all_run_data)
    
    # Export aggregated metrics
    export_aggregated_metrics(results_dir, all_run_data)
    
    print("Evaluation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
