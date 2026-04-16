#!/usr/bin/env python3
"""
LLM as Judge: Evaluate Rule-Case Relevance
Calls DeepSeek API in parallel to evaluate the relevance between rules and cases.
Scores are from 1 to 3:
  1 = No connection whatsoever: Completely unrelated to the case
  2 = Moderate relevance: Related to some aspects of the case
  3 = Highly applicable: Core rule for this case, directly applies
"""

import json
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import aiohttp
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse

@dataclass
class JudgmentResult:
    """Data class for storing judgment results"""
    case_name: str
    rule: str
    llm_response: str
    relevance_score: int
    reasoning: str
    timestamp: str
    source_file_path: Path = None  # Internal use only, not serialized
    case_dict: Dict = None  # Internal use only, not serialized


class RelevanceJudge:
    """Judge for evaluating rule-case relevance using LLM"""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat", 
                 max_workers: int = 10, api_endpoint: str = "https://api.deepseek.com/chat/completions"):
        """
        Initialize the judge
        
        Args:
            api_key: API key for the model (DeepSeek or others)
            model: Model name to use
            max_workers: Maximum number of concurrent API calls
            api_endpoint: API endpoint URL
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.max_workers = max_workers
        self.api_endpoint = api_endpoint
        self.results: List[JudgmentResult] = []
        self.semaphore = None
        
        if not self.api_key:
            raise ValueError("API key not provided. Set via argument or DEEPSEEK_API_KEY environment variable.")
        
        print(f"✓ RelevanceJudge initialized")
        print(f"  Model: {model}")
        print(f"  API Endpoint: {api_endpoint}")
        print(f"  Max Workers: {max_workers}")
    
    async def _call_llm(self, session: aiohttp.ClientSession, prompt: str) -> Tuple[str, bool]:
        """
        Call LLM API with the given prompt
        
        Args:
            session: aiohttp client session
            prompt: The prompt to send to LLM
            
        Returns:
            Tuple of (response_text, success_flag)
        """
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
            "max_tokens": 10
        }
        
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
                        result_text = data["choices"][0]["message"]["content"]
                        return result_text, True
                    else:
                        error_msg = await response.text()
                        print(f"✗ API Error: {response.status} - {error_msg}")
                        return "", False
            except asyncio.TimeoutError:
                print("✗ API call timeout")
                return "", False
            except Exception as e:
                print(f"✗ Error calling API: {str(e)}")
                return "", False
    
    def _build_prompt(self, case_background: str, source_rule: str) -> str:
        """
        Build the evaluation prompt
        
        Args:
            case_background: Background/description of the case
            source_rule: The rule being evaluated
            
        Returns:
            The prompt string
        """



#         - 0 = No connection whatsoever: Completely unrelated to the case.
# - 1 = Minimal relevance: Has some distant relationship to the case topic.
# - 2 = Moderate relevance: Related to some aspects of the case.
# - 3 = Strong relevance: Directly applicable to the case scenario.
# - 4 = Highly applicable: Core rule for this case, directly applies.

# - 0 = No connection whatsoever: Completely unrelated to the case.
# - 1 = Moderate relevance: Related to some aspects of the case.
# - 2 = Highly applicable: Core rule for this case, directly applies.
    
#     - 0 = No connection whatsoever: Completely unrelated to the case.
# - 1 = Minimal relevance: Has some distant relationship to the case topic.
# - 2 = Strong relevance: Directly applicable to the case scenario.


        prompt = f"""You are an expert in compliance and regulatory evaluation. Your task is to assess the relevance of a rule to a specific case.

Evaluate the following rule against the case background and determine the relevance score.

SCORING RUBRIC (respond with ONLY the score as a single number):
- 1 = No connection whatsoever: Completely unrelated to the case.
- 2 = Moderate relevance: Has some distant relationship to the case topic.
- 3 = Strong relevance: Directly applicable to the case scenario.

CASE BACKGROUND:
{case_background}

RULE:
{source_rule}

Respond with ONLY a single number from 1-3, nothing else."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[int, str]:
        """
        Parse LLM response to extract score
        
        Args:
            response: Raw LLM response (should be a single number 1-3)
            
        Returns:
            Tuple of (score, reasoning) where score is 1-3 or -1 if parsing fails
        """
        try:
            # Extract the first digit from the response
            response_clean = response.strip()
            
            for char in response_clean:
                if char.isdigit():
                    score = int(char)
                    if 1 <= score <= 3:
                        return score, response_clean
            
            # If no valid digit found
            return -1, response_clean
        except Exception as e:
            print(f"✗ Error parsing response: {e}")
            return -1, response
    
    async def judge_case(self, session: aiohttp.ClientSession, 
                        case_name: str, case_background: str, 
                        source_rule: str) -> JudgmentResult:
        """
        Judge a single case-rule pair
        
        Args:
            session: aiohttp client session
            case_name: Name/ID of the case
            case_background: Background of the case
            source_rule: The rule to evaluate
            
        Returns:
            JudgmentResult object
        """
        prompt = self._build_prompt(case_background, source_rule)
        response, success = await self._call_llm(session, prompt)
        
        if success:
            score, reasoning = self._parse_llm_response(response)
            print(f"✓ {case_name}: Score = {score}")
        else:
            score = -1
            reasoning = "Failed to get response from API"
            print(f"✗ {case_name}: Failed to get response")
        
        result = JudgmentResult(
            case_name=case_name,
            rule=source_rule,
            llm_response=response,
            relevance_score=score,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    async def judge_cases_parallel(self, cases_with_paths: List[Tuple[Dict[str, Any], Path]]) -> List[JudgmentResult]:
        """
        Judge multiple case-rule pairs in parallel
        
        Args:
            cases_with_paths: List of tuples (case_dict, source_file_path) where:
                   - case_dict has 'example_name', 'example_background', 'source_rule' fields
                   - source_file_path is the Path to the JSON file it came from
            
        Returns:
            List of JudgmentResult objects
        """
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for case, source_file in cases_with_paths:
                case_name = case.get("example_name", "Unknown")
                case_background = case.get("example_background", "")
                source_rule = case.get("source_rule", "")
                
                if not case_background or not source_rule:
                    print(f"⚠ Skipping {case_name}: Missing background or rule")
                    continue
                
                # Store source file path in result for later use
                task = self._judge_case_with_path(session, case_name, case_background, 
                                                 source_rule, case, source_file)
                tasks.append(task)
            
            # Run all tasks concurrently
            print(f"\n▶ Processing {len(tasks)} cases in parallel...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, JudgmentResult)]
            return valid_results
    
    async def _judge_case_with_path(self, session: aiohttp.ClientSession, 
                                   case_name: str, case_background: str, 
                                   source_rule: str, case_dict: Dict, 
                                   source_file: Path) -> JudgmentResult:
        """
        Judge a single case-rule pair and attach source file path
        
        Args:
            session: aiohttp client session
            case_name: Name/ID of the case
            case_background: Background of the case
            source_rule: The rule to evaluate
            case_dict: Full case dictionary
            source_file: Path to the source JSON file
            
        Returns:
            JudgmentResult object with source_file_path set
        """
        result = await self.judge_case(session, case_name, case_background, source_rule)
        # Attach source file path for later use in save_results
        result.source_file_path = source_file
        result.case_dict = case_dict
        return result
    
    def save_results(self, output_base_path: str, input_base_path: str = None) -> None:
        """
        Save judgment results to separate JSON files, one for each input file
        Preserves directory structure from input
        
        Args:
            output_base_path: Base path for output directory (e.g., 'results')
            input_base_path: Base path of input (for computing relative paths)
        """
        # Group results by source file
        results_by_file = defaultdict(list)
        
        for result in self.results:
            source_file = result.source_file_path if result.source_file_path else None
            results_by_file[source_file].append(result)
        
        output_base = Path(output_base_path)
        input_base = Path(input_base_path) if input_base_path else None
        
        total_saved = 0
        all_file_summaries = []
        
        # Process each input file separately
        for source_file, file_results in results_by_file.items():
            if source_file is None:
                # Fallback: save to output_base with a default name
                output_file = output_base / "results.json"
                file_path_str = "unknown"
            else:
                # Compute relative path from input base
                if input_base:
                    try:
                        # Get relative path from parent of input_base
                        relative_path = source_file.relative_to(input_base.parent)
                    except ValueError:
                        # If relative_to fails, use the filename
                        relative_path = Path(source_file.name)
                else:
                    relative_path = Path(source_file.name)
                
                file_path_str = str(relative_path)
                output_file = output_base / relative_path
            
            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert results to dictionaries for JSON serialization
            results_data = []
            for r in file_results:
                r_dict = {
                    'case_name': r.case_name,
                    'rule': r.rule,
                    'llm_response': r.llm_response,
                    'relevance_score': r.relevance_score,
                    'timestamp': r.timestamp
                }
                results_data.append(r_dict)
            
            # Calculate file-level statistics
            file_valid_scores = [r.relevance_score for r in file_results if r.relevance_score >= 0]
            
            if file_valid_scores:
                file_avg_score = sum(file_valid_scores) / len(file_valid_scores)
                # Normalize to percentage (1-3 maps to 0-100%)
                file_score_percentage = ((file_avg_score) / 3.0) * 100
            else:
                file_avg_score = -1
                file_score_percentage = -1
            
            # Create output data with metadata and results
            file_output = {
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path_str,
                'input_file': str(source_file) if source_file else None,
                'total_cases': len(file_results),
                'successful_cases': len(file_valid_scores),
                'failed_cases': len(file_results) - len(file_valid_scores),
                'average_score': round(file_avg_score, 2),
                'score_percentage': round(file_score_percentage, 2),
                'results': results_data
            }
            
            # Save to individual file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(file_output, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved {len(file_results)} results to {output_file}")
            print(f"  Average Score: {file_avg_score:.2f}/3.0 ({file_score_percentage:.2f}%)")
            
            # Add to summary list
            all_file_summaries.append({
                'file_path': file_path_str,
                'output_file': str(output_file),
                'total_cases': len(file_results),
                'average_score': round(file_avg_score, 2),
                'score_percentage': round(file_score_percentage, 2)
            })
            
            total_saved += len(file_results)
        
        # Create and save overall summary file
        overall_valid_scores = [r.relevance_score for r in self.results if r.relevance_score >= 0]
        if overall_valid_scores:
            overall_avg = sum(overall_valid_scores) / len(overall_valid_scores)
            overall_percentage = ((overall_avg) / 3.0) * 100
        else:
            overall_avg = -1
            overall_percentage = -1
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(self.results),
            'successful_cases': len(overall_valid_scores),
            'failed_cases': len(self.results) - len(overall_valid_scores),
            'overall_average_score': round(overall_avg, 2),
            'overall_score_percentage': round(overall_percentage, 2),
            'file_summaries': all_file_summaries
        }
        
        # summary_file = output_base / "_summary.json"
        # summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        # with open(summary_file, 'w', encoding='utf-8') as f:
        #     json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # print(f"\n✓ Summary saved to {summary_file}")
        print(f"✓ Total results saved: {total_saved}")
    
    def print_summary(self) -> None:
        """Print summary statistics of judgment results"""
        if not self.results:
            print("No results to summarize")
            return
        
        total = len(self.results)
        score_dist = defaultdict(int)
        failed = 0
        
        for result in self.results:
            if result.relevance_score == -1:
                failed += 1
            else:
                score_dist[result.relevance_score] += 1
        
        # Calculate overall statistics
        valid_scores = [r.relevance_score for r in self.results if r.relevance_score >= 0]
        if valid_scores:
            overall_avg = sum(valid_scores) / len(valid_scores)
            # Normalize to percentage (1-3 maps to 0-100%, where 1=0%, 3=100%)
            overall_percentage = ((overall_avg) / 3.0) * 100
        else:
            overall_avg = -1
            overall_percentage = -1
        
        print("\n" + "="*60)
        print("JUDGMENT SUMMARY")
        print("="*60)
        print(f"Total cases evaluated: {total}")
        print(f"Successfully judged: {total - failed}")
        print(f"Failed: {failed}")
        print(f"\nOverall Average Score: {overall_avg:.2f}/3.0 ({overall_percentage:.2f}%)")
        print("\nScore Distribution:")
        for score in range(1, 4):
            count = score_dist.get(score, 0)
            percentage = (count / (total - failed) * 100) if (total - failed) > 0 else 0
            print(f"  {score}: {count:3d} ({percentage:5.1f}%)")
        
        print("\nDetailed Results:")
        print("-"*60)
        for result in self.results:
            status = "✓" if result.relevance_score != -1 else "✗"
            print(f"{status} {result.case_name}: Score = {result.relevance_score}")
            print()


def load_cases_from_directory(directory: str) -> List[Tuple[Dict[str, Any], Path]]:
    """
    Load all JSON case files from a directory recursively
    
    Args:
        directory: Path to directory containing JSON files
        
    Returns:
        List of tuples (case_dict, source_file_path) to preserve directory structure
    """
    all_cases = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    # Find all JSON files recursively
    json_files = list(dir_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for case in data:
                        all_cases.append((case, json_file))
                else:
                    all_cases.append((data, json_file))
            print(f"  ✓ Loaded {len(data) if isinstance(data, list) else 1} cases from {json_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {json_file}: {e}")
    
    print(f"Total cases loaded: {len(all_cases)}\n")
    return all_cases


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LLM-based Rule-Case Relevance Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_as_judge.py \\
    --input_path data_cases_v2/cyber_security/mitre_attack \\
    --output_path results/relevance_scores.json \\
    --model deepseek-chat \\
    --max_workers 5
        """
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input directory containing JSON case files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory for results (e.g., 'results'). One JSON file per input file will be created."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name to use (default: deepseek-chat)"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://api.deepseek.com",
        help="API base URL (default: https://api.deepseek.com)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (or set DEEPSEEK_API_KEY environment variable)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of parallel API calls (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Build API endpoint
    api_endpoint = args.api_base.rstrip('/') + "/chat/completions"
    
    # Load cases
    print("Loading cases from directory...")
    cases_with_paths = load_cases_from_directory(args.input_path)
    
    if not cases_with_paths:
        print("✗ No cases found. Exiting.")
        return
    
    # Initialize judge
    try:
        judge = RelevanceJudge(
            api_key=args.api_key,
            model=args.model,
            max_workers=args.max_workers,
            api_endpoint=api_endpoint
        )
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
    
    # Judge cases in parallel
    start_time = time.time()
    try:
        results = await judge.judge_cases_parallel(cases_with_paths)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    
    elapsed_time = time.time() - start_time
    
    # Save results preserving directory structure
    judge.save_results(args.output_path, args.input_path)
    
    # Print summary
    judge.print_summary()
    
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Average time per case: {elapsed_time / len(judge.results):.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
