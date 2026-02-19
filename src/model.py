"""
Model loading and prompting utilities for LtM experiments.
"""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LtMModel:
    """Wrapper for language model with LtM prompting."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 2048,
        cache_dir: str = ".cache"
    ):
        """Initialize model and tokenizer."""
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 180,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text from prompt."""
        # Format prompt as chat message for instruct models
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()


def build_decomposition_prompt(question: str) -> str:
    """Build prompt for decomposing question into subquestions."""
    return f"""You are solving a math word problem. Break down the following problem into 3-6 numbered subquestions that need to be answered step-by-step.

Problem: {question}

Decompose this into subquestions (numbered 1-6):"""


def build_step_prompt(
    question: str,
    subquestions: List[str],
    step_idx: int,
    previous_answers: List[str],
    operator: Optional[str] = None
) -> str:
    """Build prompt for answering a single step."""
    context = f"Original problem: {question}\n\n"
    
    # Add previous subquestions and answers
    for i in range(step_idx):
        context += f"Subquestion {i+1}: {subquestions[i]}\n"
        context += f"Answer {i+1}: {previous_answers[i]}\n\n"
    
    # Current subquestion
    context += f"Subquestion {step_idx+1}: {subquestions[step_idx]}\n"
    
    # Add operator-specific instruction if provided
    if operator == "forward_algebra":
        instruction = "Solve this step algebraically, showing your work."
    elif operator == "backward_check":
        instruction = "First hypothesize a plausible answer, then verify it works backward."
    elif operator == "estimation":
        instruction = "Start by estimating reasonable bounds, then calculate the precise answer."
    elif operator == "unit_constraint":
        instruction = "Check units and constraints first, then solve."
    else:
        instruction = "Answer this subquestion based on the information above."
    
    context += f"\n{instruction}\nProvide your answer:"
    
    return context


def build_verification_prompt(
    question: str,
    subquestion: str,
    candidate_answer: str,
    previous_context: str
) -> str:
    """Build prompt for verifying a candidate answer."""
    prompt = f"""Original problem: {question}

{previous_context}

Subquestion: {subquestion}
Candidate answer: {candidate_answer}

Verify if this answer is VALID (consistent and reasonable) or INVALID (inconsistent or unreasonable).
Also provide a normalized short answer if valid.

Format your response as:
Verdict: VALID or INVALID
Normalized answer: <short answer if valid>
Reasoning: <brief explanation>"""
    
    return prompt


def build_final_prompt(
    question: str,
    subquestions: List[str],
    subanswers: List[str]
) -> str:
    """Build prompt for generating final answer."""
    context = f"Original problem: {question}\n\n"
    
    # Add all subquestions and answers
    for i in range(len(subquestions)):
        context += f"Subquestion {i+1}: {subquestions[i]}\n"
        context += f"Answer {i+1}: {subanswers[i]}\n\n"
    
    context += "Based on the above, provide the final numeric answer.\n"
    context += "Format: Final Answer: <number>"
    
    return context
