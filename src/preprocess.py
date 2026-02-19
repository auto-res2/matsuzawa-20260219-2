"""
Dataset loading and preprocessing for GSM8K experiments.
"""
import re
from typing import Dict, List, Any, Optional
from datasets import load_dataset


def load_gsm8k(
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
    cache_dir: str = ".cache"
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset and prepare for inference.
    
    Args:
        split: Dataset split (train/test)
        n_samples: Number of samples to load (None for all)
        seed: Random seed for sampling
        cache_dir: Cache directory for datasets
    
    Returns:
        List of examples with 'question' and 'answer' keys
    """
    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    
    # Shuffle and sample if needed
    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    
    # Process examples
    examples = []
    for item in dataset:
        question = item["question"]
        # Extract numeric answer from solution (format: #### <number>)
        answer_text = item["answer"]
        answer_match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_text)
        if answer_match:
            # Remove commas from numbers
            answer = float(answer_match.group(1).replace(",", ""))
        else:
            # Fallback: try to extract any number at the end
            numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", answer_text)
            if numbers:
                answer = float(numbers[-1].replace(",", ""))
            else:
                answer = None
        
        examples.append({
            "question": question,
            "answer": answer,
            "answer_text": answer_text
        })
    
    return examples


def extract_final_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from model output.
    Looks for patterns like "Final Answer: 123" or just numbers.
    
    Args:
        text: Model output text
    
    Returns:
        Extracted number or None
    """
    # Try to find "Final Answer: <number>" pattern
    match = re.search(r"(?:Final Answer|Answer|Result):\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Try to find any number at the end
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    
    return None


def canonicalize_answer(text: str) -> str:
    """
    Canonicalize answer text for voting.
    Normalizes numbers, whitespace, and common variations.
    
    Args:
        text: Answer text to canonicalize
    
    Returns:
        Canonicalized string
    """
    # Extract numbers and normalize
    text = text.lower().strip()
    # Remove commas from numbers
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text
