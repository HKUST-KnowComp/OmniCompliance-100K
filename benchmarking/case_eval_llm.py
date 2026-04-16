#!/usr/bin/env python3
"""
LLM Evaluation Script for Academic Integrity Cases
Calls DeepSeek API in parallel to evaluate case compliance
"""

import json
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import aiohttp
from dataclasses import dataclass
from collections import defaultdict

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import argparse
parser = argparse.ArgumentParser(description="LLM Case Evaluation")
parser.add_argument("--input_file_path", type=str, default='../data/cases/edu/academic_integrity', help="input data directory")
parser.add_argument("--output_file_path", type=str, default='./results_path', help="output directory")
parser.add_argument("--model_type", type=str, default="api", choices=["api", "local"], help="model type: api or local")
parser.add_argument("--model_name", type=str, default="deepseek-chat", help="model name (for API) or model path (for local)")
parser.add_argument("--api_endpoint", type=str, default="https://api.deepseek.com/chat/completions", help="endpoint for api calling")
parser.add_argument("--max_workers", type=int, default=10, help="number of concurrent workers")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for local models")
parser.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "huggingface"], help="backend for local models: auto (try vLLM first), vllm, or huggingface")
parser.add_argument("--batch_mode", action="store_true", help="enable batch processing for local models")
parser.add_argument("--batch_size", type=int, default=8, help="batch size for batch processing")
args = parser.parse_args()

@dataclass
class EvaluationResult:
    """Data class for storing evaluation results"""
    case_name: str
    ground_truth: str
    llm_response: str
    llm_prediction: str
    is_correct: bool
    reasoning: str
    timestamp: str


class LocalLLMInference:
    """Local LLM inference using vLLM or HuggingFace"""
    
    def __init__(self, model_name: str, temperature: float = 0.3, max_tokens: int = 100, 
                 gpu_memory_utilization: float = 0.9, backend: str = "auto"):
        """
        Initialize local LLM model
        
        Args:
            model_name: Model name or path (e.g., "meta-llama/Llama-2-7b-hf")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: GPU memory utilization ratio (vLLM only)
            backend: "vllm", "huggingface", or "auto" (try vLLM first, fall back to HF)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = backend
        self.llm = None
        
        print(f"Loading model: {model_name}")
        print(f"Backend: {backend}")
        
        # Try vLLM first if backend is "auto" or "vllm"
        if (backend in ["auto", "vllm"]) and HAS_VLLM:
            try:
                self.llm = self._init_vllm(model_name, gpu_memory_utilization)
                self.backend = "vllm"
                return
            except Exception as e:
                if backend == "vllm":
                    print(f"✗ Error loading with vLLM: {e}")
                    raise
                print(f"⚠ vLLM failed: {e}, falling back to HuggingFace")
        
        # Fall back to HuggingFace
        if backend in ["auto", "huggingface"] and HAS_TRANSFORMERS:
            self.llm = self._init_huggingface(model_name)
            self.backend = "huggingface"
        else:
            raise ImportError("Neither vLLM nor HuggingFace transformers available")
    
    def _init_vllm(self, model_name: str, gpu_memory_utilization: float) -> LLM:
        """Initialize vLLM backend"""
        print(f"Initializing vLLM...")
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
            trust_remote_code=True,
            tensor_parallel_size=1
        )
        print(f"✓ vLLM model loaded: {model_name}")
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95
        )
        return llm
    
    def _init_huggingface(self, model_name: str):
        """Initialize HuggingFace backend"""
        print(f"Initializing HuggingFace pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ HuggingFace model loaded: {model_name}")
        return pipe
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using the loaded model
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            if self.backend == "vllm":
                outputs = self.llm.generate([prompt], self.sampling_params)
                result_text = outputs[0].outputs[0].text.strip()
            else:  # huggingface
                outputs = self.llm(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    return_full_text=False
                )
                result_text = outputs[0]["generated_text"].strip()
            
            return result_text
        except Exception as e:
            print(f"Error during inference: {e}")
            return ""
    
    async def generate_async(self, prompt: str) -> Tuple[str, bool]:
        """
        Async wrapper for generation
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (generated_text, success_flag)
        """
        try:
            result = await asyncio.to_thread(self.generate, prompt)
            success = len(result) > 0
            return result, success
        except Exception as e:
            print(f"Error in async generation: {e}")
            return "", False
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate text for multiple prompts (batch processing)
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated texts
        """
        try:
            if self.backend == "vllm":
                # vLLM handles batching efficiently
                outputs = self.llm.generate(prompts, self.sampling_params)
                results = [outputs[i].outputs[0].text.strip() for i in range(len(outputs))]
            else:  # huggingface
                # HuggingFace pipeline batch processing
                results = []
                for prompt in prompts:
                    try:
                        output = self.llm(
                            prompt,
                            max_new_tokens=self.max_tokens,
                            temperature=self.temperature,
                            do_sample=True,
                            top_p=0.95,
                            return_full_text=False
                        )
                        results.append(output[0]["generated_text"].strip())
                    except Exception as e:
                        print(f"Error processing prompt: {e}")
                        results.append("")
            
            return results
        except Exception as e:
            print(f"Error during batch inference: {e}")
            return [""] * len(prompts)
    
    async def generate_batch_async(self, prompts: List[str]) -> Tuple[List[str], bool]:
        """
        Async wrapper for batch generation
        
        Args:
            prompts: List of input prompts
            
        Returns:
            Tuple of (list_of_generated_texts, success_flag)
        """
        try:
            results = await asyncio.to_thread(self.generate_batch, prompts)
            success = any(len(r) > 0 for r in results)
            return results, success
        except Exception as e:
            print(f"Error in async batch generation: {e}")
            return [""] * len(prompts), False


class CaseEvaluator:
    """Main evaluator class for LLM-based case evaluation"""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat", max_workers: int = 5, 
                 model_type: str = "api", local_llm = None, gpu_memory_utilization: float = 0.9):
        """
        Initialize the evaluator
        
        Args:
            api_key: DeepSeek API key (for API models)
            model: Model name to use
            max_workers: Maximum number of concurrent API calls
            model_type: "api" or "local"
            local_llm: LocalLLMInference instance (for local models)
            gpu_memory_utilization: GPU memory utilization for local models
        """
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.model_type = model_type
        self.local_llm = local_llm
        self.gpu_memory_utilization = gpu_memory_utilization
        
        if model_type == "api":
            self.api_endpoint = args.api_endpoint
            if 'deepseek' in args.model_name:
                self.api_endpoint = "https://api.deepseek.com/chat/completions"
            print(self.api_endpoint)
        elif model_type == "local" and local_llm is None:
            self.local_llm = LocalLLMInference(model, gpu_memory_utilization=gpu_memory_utilization)
        
        self.results: List[EvaluationResult] = []
        self.semaphore = None
        
    async def _call_llm(self, session: aiohttp.ClientSession, prompt: str) -> Tuple[str, bool]:
        """
        Call LLM (API or local) with the given prompt
        
        Args:
            session: aiohttp client session (only used for API)
            prompt: The prompt to send to LLM
            
        Returns:
            Tuple of (response_text, success_flag)
        """
        if self.model_type == "local":
            return await self.local_llm.generate_async(prompt)
        
        # API call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 100
        }
        if 'grok' in args.model_name:
            payload['input'] = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        
        async with self.semaphore:
            try:
                async with session.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'grok' in args.model_name:
                            result_text = data['output'][0]['content'][0]['text']
                        else:
                            result_text = data["choices"][0]["message"]["content"]
                        # print('here', data)
                        return result_text, True
                    else:
                        error_msg = await response.text()
                        print(f"API Error: {response.status} - {error_msg}")
                        return "", False
            except asyncio.TimeoutError:
                print("API call timeout")
                return "", False
            except Exception as e:
                print(f"Error calling API: {str(e)}")
                return "", False
    
    def _build_prompt(self, case_background: str, source_rule: str) -> str:
        """
        Build the evaluation prompt
        
        Args:
            case_background: Background of the case
            source_rule: The rule being evaluated
            
        Returns:
            The prompt string
        """
        prompt = f"""You are an expert in compliance evaluation.

Based on the following case background, determine if the case represents PERMITTED or PROHIBITED behavior.

Case Background:
{case_background}

Answer with ONLY one word: either "PERMITTED" or "PROHIBITED". Do not provide any explanation, just the single word answer."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> str:
        """
        Parse LLM response to extract prediction
        
        Args:
            response: Raw LLM response
            
        Returns:
            Either "PERMITTED" or "PROHIBITED" or "UNKNOWN" if parsing fails
        """
        response_clean = response.strip().upper()
        
        # Check for PERMITTED
        if "PERMITTED" in response_clean:
            return "PERMITTED"
        # Check for PROHIBITED
        elif "PROHIBITED" in response_clean:
            return "PROHIBITED"
        # Check for VIOLATES (equivalent to PROHIBITED)
        elif "VIOLATES" in response_clean:
            return "PROHIBITED"
        # Check for COMPLIES (equivalent to PERMITTED)
        elif "COMPLIES" in response_clean or "COMPLIANT" in response_clean:
            return "PERMITTED"
        else:
            return "UNKNOWN"
    
    def _construct_prompt(self, case: Dict[str, Any]) -> str:
        """
        Construct prompt from case dictionary for batch processing
        
        Args:
            case: Case dictionary with example_background and source_rule
            
        Returns:
            The prompt string
        """
        background = case.get("example_background", "")
        source_rule = case.get("source_rule", "")
        
        prompt = f"""You are an expert in compliance evaluation.

Based on the following case background and rule, determine if the case represents PERMITTED or PROHIBITED behavior.

Case Background:
{background}

Answer with ONLY one word: either "PERMITTED" or "PROHIBITED". Do not provide any explanation, just the single word answer."""
        
        return prompt
    
    def _extract_prediction(self, response: str) -> str:
        """
        Extract prediction from LLM response
        
        Args:
            response: Raw LLM response
            
        Returns:
            Either "PERMITTED" or "PROHIBITED" or "UNKNOWN"
        """
        return self._parse_llm_response(response)
    
    def _normalize_ground_truth(self, relation: str) -> str:
        """
        Normalize ground truth to standard format
        
        Args:
            relation: The relation_to_rule value
            
        Returns:
            Either "PERMITTED" or "PROHIBITED"
        """
        relation_upper = relation.strip().upper()
        
        if relation_upper in ["COMPLIES", "COMPLIANT", "PERMITTED"]:
            return "PERMITTED"
        elif relation_upper in ["VIOLATES", "PROHIBITED", "VIOLATION"]:
            return "PROHIBITED"
        else:
            return relation_upper
    
    async def evaluate_case(self, session: aiohttp.ClientSession, case: Dict[str, Any], 
                           case_idx: int, total_cases: int) -> EvaluationResult:
        """
        Evaluate a single case
        
        Args:
            session: aiohttp client session
            case: Case dictionary
            case_idx: Index of the case
            total_cases: Total number of cases
            
        Returns:
            EvaluationResult object
        """
        case_name = case.get("example_name", f"Case_{case_idx}")
        background = case.get("example_background", "")
        source_rule = case.get("source_rule", "")
        ground_truth_raw = case.get("relation_to_rule", "UNKNOWN")
        ground_truth = self._normalize_ground_truth(ground_truth_raw)
        
        # Build and send prompt
        prompt = self._build_prompt(background, source_rule)
        llm_response, success = await self._call_llm(session, prompt)
        
        if not success:
            llm_response = "API_ERROR"
        
        # Parse response
        llm_prediction = self._parse_llm_response(llm_response)
        
        # Check correctness
        is_correct = llm_prediction == ground_truth and llm_prediction != "UNKNOWN"
        
        # Create result
        result = EvaluationResult(
            case_name=case_name,
            ground_truth=ground_truth,
            llm_response=llm_response[:200],  # Truncate for storage
            llm_prediction=llm_prediction,
            is_correct=is_correct,
            reasoning=f"Expected: {ground_truth}, Got: {llm_prediction}",
            timestamp=datetime.now().isoformat()
        )
        
        # Print progress
        status = "✓" if is_correct else "✗"
        print(f"[{case_idx + 1}/{total_cases}] {status} {case_name[:50]:50} | "
              f"Truth: {ground_truth:10} | Prediction: {llm_prediction:10}")
        
        return result
    
    async def evaluate_cases_batch(self, cases: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate multiple cases using batch processing (optimized for local models)
        
        Args:
            cases: List of case dictionaries
            batch_size: Number of cases to process in each batch
            
        Returns:
            Dictionary with evaluation statistics
        """
        if self.model_type != "local":
            print("⚠ Batch processing is optimized for local models. Using standard evaluation.")
            return await self.evaluate_cases(cases)
        
        if not self.local_llm:
            raise ValueError("Local LLM not initialized")
        
        self.results = []
        total_cases = len(cases)
        
        print(f"Processing {total_cases} cases in batches of {batch_size}")
        
        for batch_idx in range(0, total_cases, batch_size):
            batch_cases = cases[batch_idx:batch_idx + batch_size]
            batch_start = time.time()
            
            # Prepare prompts for the batch
            prompts = []
            case_refs = []
            for case in batch_cases:
                prompt = self._construct_prompt(case)
                prompts.append(prompt)
                case_refs.append(case)
            
            # Batch inference
            print(f"\n[Batch {batch_idx // batch_size + 1}] Processing {len(batch_cases)} cases...")
            responses, success = await self.local_llm.generate_batch_async(prompts)
            
            # Process responses
            for idx, (case, response) in enumerate(zip(case_refs, responses)):
                case_idx = batch_idx + idx
                ground_truth = case.get("relation_to_rule", "").upper()
                
                # Normalize ground truth
                if "COMPLIES" in ground_truth or "PERMITTED" in ground_truth:
                    ground_truth = "PERMITTED"
                elif "VIOLATES" in ground_truth or "PROHIBITED" in ground_truth:
                    ground_truth = "PROHIBITED"
                
                # Extract prediction
                llm_prediction = self._extract_prediction(response)
                
                # Normalize prediction
                if any(word in llm_prediction.upper() for word in ["PERMITTED", "COMPLIES", "COMPLIANT"]):
                    llm_prediction = "PERMITTED"
                elif any(word in llm_prediction.upper() for word in ["PROHIBITED", "VIOLATES"]):
                    llm_prediction = "PROHIBITED"
                else:
                    llm_prediction = "UNKNOWN"
                
                is_correct = (ground_truth == llm_prediction) and (llm_prediction != "UNKNOWN")
                case_name = case.get("example_name", "Unknown")
                
                result = EvaluationResult(
                    case_name=case_name,
                    ground_truth=ground_truth,
                    llm_response=response,
                    llm_prediction=llm_prediction,
                    is_correct=is_correct,
                    reasoning=case.get("source_rule", ""),
                    timestamp=datetime.now().isoformat()
                )
                self.results.append(result)
                
                # Print progress
                status = "✓" if is_correct else "✗"
                print(f"  [{case_idx + 1}/{total_cases}] {status} {case_name[:40]:40} | "
                      f"Truth: {ground_truth:10} | Pred: {llm_prediction:10}")
            
            batch_elapsed = time.time() - batch_start
            batch_throughput = len(batch_cases) / batch_elapsed if batch_elapsed > 0 else 0
            print(f"  Batch completed in {batch_elapsed:.2f}s ({batch_throughput:.2f} cases/s)")
        
        stats = self._compute_statistics()
        return stats
    
    async def evaluate_cases(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple cases in parallel
        
        Args:
            cases: List of case dictionaries
            
        Returns:
            Dictionary with evaluation statistics
        """
        self.semaphore = asyncio.Semaphore(self.max_workers)
        self.results = []
        
        if self.model_type == "local":
            # Local model - no session needed
            tasks = [
                self.evaluate_case(None, case, idx, len(cases))
                for idx, case in enumerate(cases)
            ]
            self.results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # API model - use session
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.evaluate_case(session, case, idx, len(cases))
                    for idx, case in enumerate(cases)
                ]
                self.results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_results = []
        for result in self.results:
            if isinstance(result, Exception):
                print(f"Error in evaluation: {result}")
            else:
                valid_results.append(result)
        
        self.results = valid_results
        stats = self._compute_statistics()
        return stats
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """
        Compute evaluation statistics including F1 score
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Breakdown by ground truth
        permitted_cases = [r for r in self.results if r.ground_truth == "PERMITTED"]
        prohibited_cases = [r for r in self.results if r.ground_truth == "PROHIBITED"]
        
        permitted_correct = sum(1 for r in permitted_cases if r.is_correct)
        prohibited_correct = sum(1 for r in prohibited_cases if r.is_correct)
        
        permitted_acc = (permitted_correct / len(permitted_cases) * 100) if permitted_cases else 0
        prohibited_acc = (prohibited_correct / len(prohibited_cases) * 100) if prohibited_cases else 0
        
        # Calculate F1 scores (treating PERMITTED as positive class)
        # tp: correctly predicted as PERMITTED
        tp_permitted = sum(1 for r in self.results if r.ground_truth == "PERMITTED" and r.llm_prediction == "PERMITTED")
        # fp: incorrectly predicted as PERMITTED (should be PROHIBITED)
        fp_permitted = sum(1 for r in self.results if r.ground_truth == "PROHIBITED" and r.llm_prediction == "PERMITTED")
        # fn: incorrectly predicted as not PERMITTED (should be PERMITTED)
        fn_permitted = sum(1 for r in self.results if r.ground_truth == "PERMITTED" and r.llm_prediction != "PERMITTED")
        
        # tp: correctly predicted as PROHIBITED
        tp_prohibited = sum(1 for r in self.results if r.ground_truth == "PROHIBITED" and r.llm_prediction == "PROHIBITED")
        # fp: incorrectly predicted as PROHIBITED (should be PERMITTED)
        fp_prohibited = sum(1 for r in self.results if r.ground_truth == "PERMITTED" and r.llm_prediction == "PROHIBITED")
        # fn: incorrectly predicted as not PROHIBITED (should be PROHIBITED)
        fn_prohibited = sum(1 for r in self.results if r.ground_truth == "PROHIBITED" and r.llm_prediction != "PROHIBITED")
        
        # F1 for PERMITTED
        precision_permitted = tp_permitted / (tp_permitted + fp_permitted) if (tp_permitted + fp_permitted) > 0 else 0
        recall_permitted = tp_permitted / (tp_permitted + fn_permitted) if (tp_permitted + fn_permitted) > 0 else 0
        f1_permitted = 2 * (precision_permitted * recall_permitted) / (precision_permitted + recall_permitted) if (precision_permitted + recall_permitted) > 0 else 0
        
        # F1 for PROHIBITED
        precision_prohibited = tp_prohibited / (tp_prohibited + fp_prohibited) if (tp_prohibited + fp_prohibited) > 0 else 0
        recall_prohibited = tp_prohibited / (tp_prohibited + fn_prohibited) if (tp_prohibited + fn_prohibited) > 0 else 0
        f1_prohibited = 2 * (precision_prohibited * recall_prohibited) / (precision_prohibited + recall_prohibited) if (precision_prohibited + recall_prohibited) > 0 else 0
        
        # Macro F1 (only if there are valid predictions)
        valid_predictions = sum(1 for r in self.results if r.llm_prediction != "UNKNOWN")
        if valid_predictions > 0:
            macro_f1 = (f1_permitted + f1_prohibited) / 2
            weighted_f1 = (f1_permitted * len(permitted_cases) + f1_prohibited * len(prohibited_cases)) / total if total > 0 else 0
        else:
            macro_f1 = 0
            weighted_f1 = 0
        
        stats = {
            "total_cases": total,
            "correct": correct,
            "accuracy": round(accuracy, 2),
            "permitted_cases": len(permitted_cases),
            "permitted_correct": permitted_correct,
            "permitted_accuracy": round(permitted_acc, 2),
            "prohibited_cases": len(prohibited_cases),
            "prohibited_correct": prohibited_correct,
            "prohibited_accuracy": round(prohibited_acc, 2),
            "f1_permitted": round(f1_permitted, 4),
            "f1_prohibited": round(f1_prohibited, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "precision_permitted": round(precision_permitted, 4),
            "recall_permitted": round(recall_permitted, 4),
            "precision_prohibited": round(precision_prohibited, 4),
            "recall_prohibited": round(recall_prohibited, 4),
            "valid_predictions": valid_predictions,
            "results": self.results
        }
        
        return stats
    
    def save_results(self, output_file: str) -> None:
        """
        Save results to JSON file
        
        Args:
            output_file: Path to output file
        """
        # Compute statistics for output
        stats = self._compute_statistics()
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "total_cases": len(self.results),
            "correct": sum(1 for r in self.results if r.is_correct),
            "accuracy": round(sum(1 for r in self.results if r.is_correct) / len(self.results) * 100, 2) if self.results else 0,
            "macro_f1": stats.get("macro_f1", 0),
            "weighted_f1": stats.get("weighted_f1", 0),
            "f1_permitted": stats.get("f1_permitted", 0),
            "f1_prohibited": stats.get("f1_prohibited", 0),
            "results": [
                {
                    "case_name": r.case_name,
                    "ground_truth": r.ground_truth,
                    "llm_prediction": r.llm_prediction,
                    "is_correct": r.is_correct,
                    "llm_response": r.llm_response,
                    "reasoning": r.reasoning,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_file}")


def load_cases_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load cases from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of case dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    # Filter out incomplete cases (those with only timestamp)
    cases = [c for c in cases if len(c) > 1]
    
    return cases


def load_cases_from_directory(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load cases from all JSON files in a directory, grouped by filename
    
    Args:
        directory: Path to directory
        
    Returns:
        Dictionary with filename as key and list of cases as value
    """
    all_cases = {}
    
    for json_file in sorted(Path(directory).glob("*.json")):
        try:
            cases = load_cases_from_json(str(json_file))
            filename = json_file.stem
            all_cases[filename] = cases
            print(f"Loaded {len(cases)} cases from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_cases


def print_summary(stats: Dict[str, Any], elapsed_time: float) -> None:
    """
    Print evaluation summary with F1 scores
    
    Args:
        stats: Statistics dictionary
        elapsed_time: Time elapsed in seconds
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Cases:        {stats['total_cases']}")
    print(f"Correct:            {stats['correct']}/{stats['total_cases']}")
    print(f"Overall Accuracy:   {stats['accuracy']:.2f}%")
    print(f"Valid Predictions:  {stats.get('valid_predictions', 0)}")
    print(f"Macro F1:           {stats['macro_f1']:.4f}")
    print(f"Weighted F1:        {stats['weighted_f1']:.4f}")
    
    print("\n--- PERMITTED Cases ---")
    print(f"Count:              {stats['permitted_cases']}")
    print(f"Correct:            {stats['permitted_correct']}/{stats['permitted_cases']}")
    print(f"Accuracy:           {stats['permitted_accuracy']:.2f}%")
    print(f"Precision:          {stats['precision_permitted']:.4f}")
    print(f"Recall:             {stats['recall_permitted']:.4f}")
    print(f"F1:                 {stats['f1_permitted']:.4f}")
    
    print("\n--- PROHIBITED Cases ---")
    print(f"Count:              {stats['prohibited_cases']}")
    print(f"Correct:            {stats['prohibited_correct']}/{stats['prohibited_cases']}")
    print(f"Accuracy:           {stats['prohibited_accuracy']:.2f}%")
    print(f"Precision:          {stats['precision_prohibited']:.4f}")
    print(f"Recall:             {stats['recall_prohibited']:.4f}")
    print(f"F1:                 {stats['f1_prohibited']:.4f}")
    
    print("\n" + "-" * 80)
    print(f"Total Time:         {elapsed_time:.2f} seconds")
    if stats['total_cases'] > 0:
        print(f"Time per Case:      {elapsed_time / stats['total_cases']:.2f} seconds")
    print("=" * 80 + "\n")


def compute_file_stats(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Compute statistics for a single file's results
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with statistics
    """
    if not results:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0,
            "macro_f1": 0,
            "weighted_f1": 0
        }
    
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    permitted_cases = [r for r in results if r.ground_truth == "PERMITTED"]
    prohibited_cases = [r for r in results if r.ground_truth == "PROHIBITED"]
    
    # Calculate F1 scores
    tp_permitted = sum(1 for r in results if r.ground_truth == "PERMITTED" and r.llm_prediction == "PERMITTED")
    fp_permitted = sum(1 for r in results if r.ground_truth == "PROHIBITED" and r.llm_prediction == "PERMITTED")
    fn_permitted = sum(1 for r in results if r.ground_truth == "PERMITTED" and r.llm_prediction != "PERMITTED")
    
    tp_prohibited = sum(1 for r in results if r.ground_truth == "PROHIBITED" and r.llm_prediction == "PROHIBITED")
    fp_prohibited = sum(1 for r in results if r.ground_truth == "PERMITTED" and r.llm_prediction == "PROHIBITED")
    fn_prohibited = sum(1 for r in results if r.ground_truth == "PROHIBITED" and r.llm_prediction != "PROHIBITED")
    
    precision_permitted = tp_permitted / (tp_permitted + fp_permitted) if (tp_permitted + fp_permitted) > 0 else 0
    recall_permitted = tp_permitted / (tp_permitted + fn_permitted) if (tp_permitted + fn_permitted) > 0 else 0
    f1_permitted = 2 * (precision_permitted * recall_permitted) / (precision_permitted + recall_permitted) if (precision_permitted + recall_permitted) > 0 else 0
    
    precision_prohibited = tp_prohibited / (tp_prohibited + fp_prohibited) if (tp_prohibited + fp_prohibited) > 0 else 0
    recall_prohibited = tp_prohibited / (tp_prohibited + fn_prohibited) if (tp_prohibited + fn_prohibited) > 0 else 0
    f1_prohibited = 2 * (precision_prohibited * recall_prohibited) / (precision_prohibited + recall_prohibited) if (precision_prohibited + recall_prohibited) > 0 else 0
    
    macro_f1 = (f1_permitted + f1_prohibited) / 2
    weighted_f1 = (f1_permitted * len(permitted_cases) + f1_prohibited * len(prohibited_cases)) / total if total > 0 else 0
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "f1_permitted": round(f1_permitted, 4),
        "f1_prohibited": round(f1_prohibited, 4)
    }


async def main():
    """Main execution function"""
    
    # Configuration
    data_dir = args.input_file_path
    output_dir = args.output_file_path
    model_type = args.model_type
    model_name = args.model_name
    max_workers = args.max_workers
    gpu_memory_utilization = args.gpu_memory_utilization
    
    # Initialize model based on type
    print(f"Model Type: {model_type}")
    print(f"Model Name: {model_name}")
    print(f"Max Workers: {max_workers}\n")
    
    if model_type == "api":
        from config import ds_api_key, xinhuo_key, xai_key, qwen_key, zhipu_key, open_router_key
        if 'deepseek' in args.model_name.lower():
            api_key = os.getenv("DEEPSEEK_API_KEY", ds_api_key)
        elif 'grok' in args.model_name.lower():
            api_key = os.getenv("XAI_API_KEY", xai_key)
        elif 'qwen' in args.model_name.lower():
            api_key = os.getenv("QWEN_API_KEY", qwen_key)
        elif 'glm' in args.model_name.lower():
            api_key = os.getenv("ZHIPU_API_KEY", zhipu_key)
        elif 'oss' in args.model_name.lower():
            api_key = os.getenv("OPENROUTER_API_KEY", open_router_key)
        else:
            api_key = xinhuo_key
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        evaluator = CaseEvaluator(
            api_key=api_key, 
            model=model_name, 
            max_workers=max_workers,
            model_type="api"
        )
    else:
        # Local model
        try:
            local_llm = LocalLLMInference(
                model_name, 
                gpu_memory_utilization=gpu_memory_utilization,
                backend=args.backend
            )
            evaluator = CaseEvaluator(
                model=model_name,
                max_workers=1,  # Local models run sequentially
                model_type="local",
                local_llm=local_llm,
                gpu_memory_utilization=gpu_memory_utilization
            )
        except ImportError:
            print("Error: Required dependencies not installed.")
            print("Install vLLM with: pip install vllm")
            print("Or install HuggingFace with: pip install transformers torch")
            return
        except Exception as e:
            print(f"Error loading local model: {e}")
            return
    
    # Load cases grouped by file
    print(f"Loading cases from {data_dir}...")
    cases_by_file = load_cases_from_directory(data_dir)
    
    if not cases_by_file:
        print("No cases found!")
        return
    
    total_cases_all = sum(len(cases) for cases in cases_by_file.values())
    print(f"Total cases loaded: {total_cases_all}\n")
    
    all_results = []
    start_time = time.time()
    
    # Process each file
    for filename, cases in cases_by_file.items():
        print(f"\n{'='*80}")
        print(f"Processing: {filename} ({len(cases)} cases)")
        if args.batch_mode and model_type == "local":
            print(f"Mode: Batch Processing (batch_size={args.batch_size})")
        else:
            print(f"Mode: Sequential Processing")
        print(f"{'='*80}\n")
        
        file_start = time.time()
        
        # Evaluate cases for this file
        if args.batch_mode and model_type == "local":
            # Use batch processing for local models
            stats = await evaluator.evaluate_cases_batch(cases, batch_size=args.batch_size)
        else:
            # Standard sequential evaluation
            stats = await evaluator.evaluate_cases(cases)
        
        file_elapsed = time.time() - file_start
        
        # Print file-specific results
        print(f"\n{'-'*80}")
        print(f"Results for: {filename}")
        print(f"{'-'*80}")
        print(f"Total Cases:     {stats['total_cases']}")
        print(f"Correct:         {stats['correct']}/{stats['total_cases']}")
        print(f"Accuracy:        {stats['accuracy']:.2f}%")
        print(f"Valid Predictions: {stats.get('valid_predictions', 0)}")
        print(f"Macro F1:        {stats['macro_f1']:.4f}")
        print(f"Weighted F1:     {stats['weighted_f1']:.4f}")
        print(f"F1 (PERMITTED):  {stats['f1_permitted']:.4f}")
        print(f"F1 (PROHIBITED): {stats['f1_prohibited']:.4f}")
        print(f"Time:            {file_elapsed:.2f}s")
        print(f"{'-'*80}\n")
        
        # Save results for this file
        output_file = os.path.join(output_dir, f"{filename}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        evaluator.save_results(output_file)
        
        all_results.extend(evaluator.results)
    
    elapsed_time = time.time() - start_time
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Number of files processed: {len(cases_by_file)}")
    for filename in cases_by_file.keys():
        print(f"  - {filename}")
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
