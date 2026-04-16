
"""
AI Agent for Compliance Case Research and Analysis
Searches for relevant cases based on compliance rules, extracts detailed case information,
and saves results incrementally to JSON file.
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from xai_sdk import Client
from xai_sdk.chat import user, system

import argparse

parser = argparse.ArgumentParser(description="file paths")
parser.add_argument("--rule_file_path", type=str, default='rules.json', help="get path for rules.")
parser.add_argument("--output_file_path", type=str, default='output_path', help="output path")
parser.add_argument("--rule_catagory", type=str, default='', help="config catagory. e.g. (education - online learning)")
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMProvider:
    """Base LLM Provider for case research and analysis"""
    
    def __init__(self, api_key: str, provider: str = "deepseek"):
        self.api_key = api_key
        self.provider = provider
        
        if provider == "deepseek":
            self.base_url = 'https://api.deepseek.com/v1/chat/completions'
            self.model = "deepseek-chat"
        elif provider == 'grok':
            self.xai_client = Client(api_key=api_key)
            self.model = "grok-4-1-fast-non-reasoning"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def search_and_analyze_cases(self, rule: str, query: str = "") -> List[Dict[str, Any]]:
        """Single LLM API call to search for and analyze relevant cases based on a compliance rule
        
        Args:
            rule: The compliance rule to find cases for
            query: Additional query context
            
        Returns:
            List of case dictionaries with detailed information
        """
        # Retry parameters
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Build query context
                query_context = ""
                if query:
                    query_context = f"\nQUERY CONTEXT:\n{query}\n"
                catagory = args.rule_catagory
                prompt = f"""You are a compliance analyst. Find 8-10 real-world examples (such as company incidents, regulatory enforcement actions, fines, investigations, or case studies) related to this compliance rule or policy and return them as a JSON array.

COMPLIANCE RULE OR POLICY:
{rule} {catagory}

These examples can come from regulatory violations, compliance successes, corporate scandals, enforcement cases, or industry best/worst practices—not limited to formal court cases.

For each example, provide a JSON object with these fields:
- example_name: Name of the company, incident, or enforcement action (e.g., "Westpac AML Violations" or "Tesla Environmental Compliance")
- example_background: Background and context of the incident or practice. Please remain all the details. The description should be comprehensive and long. Should not appear words for easily distinguishing VIOLATE or COMPLIES, e.g. attack, malicious. Words should be neutral.
- process_and_outcome: Investigations, proceedings, decisions, or results.
- involved_parties: Key entities and their roles (array, e.g., company, regulator)
- applicable_regulations_or_policies: Specific laws, regulations, standards, or internal policies referenced (array)
- reference_link: URL or credible source reference (if known; otherwise a reliable search term or description)
- relation_to_rule: "VIOLATES" if the example shows breach or failure, "COMPLIES" if it demonstrates adherence or best practice

Prioritize well-documented, verifiable examples from reputable sources. If no exact matches, find closely analogous ones.

Return ONLY a valid JSON array, no additional text or markdown.

Example:
[
  {{
    "example_name": "Westpac Bank AML Breaches",
    "example_background": "Westpac failed to report millions of international transactions, enabling potential child exploitation risks...",
    "process_and_outcome": "Australian regulators investigated and imposed a record fine. The bank agreed to pay A$1.3 billion and implement remediation...",
    "involved_parties": ["Westpac Banking Corporation", "AUSTRAC (regulator)"],
    "applicable_regulations_or_policies": ["Anti-Money Laundering and Counter-Terrorism Financing Act"],
    "reference_link": "https://www.austrac.gov.au/news/media-release/westpac-pay-13b-penalty",
    "relation_to_rule": "VIOLATES"
  }},
  {{
    "example_name": "Tesla Sustainable Manufacturing Practices",
    "example_background": "Tesla implemented energy-efficient factories to meet environmental standards...",
    "process_and_outcome": "Through renewable energy use and waste reduction, Tesla exceeded requirements and avoided penalties...",
    "involved_parties": ["Tesla Inc.", "Environmental regulators"],
    "applicable_regulations_or_policies": ["EPA standards", "California environmental regulations"],
    "reference_link": "https://www.tesla.com/sustainability",
    "relation_to_rule": "COMPLIES"
  }}
]"""
                
                logger.info(f"Calling {self.provider} API to search for cases (attempt {attempt + 1}/{max_retries})...")
                
                if self.provider == "grok":
                    return self._call_grok(prompt, attempt, max_retries, retry_delay)
                else:
                    return self._call_deepseek(prompt, attempt, max_retries, retry_delay)
                    
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return [{"error": str(e), "status": "failed"}]
    
    def _call_grok(self, prompt: str, attempt: int, max_retries: int, retry_delay: int) -> List[Dict[str, Any]]:
        """Call Grok API using xai_sdk (new Client API) and capture full response including metadata"""
        try:
            # Create chat session using new Client API
            chat = self.xai_client.chat.create(model=self.model)
            
            # Append system message
            chat.append(system("You are a legal compliance analyst specializing in finding and analyzing real compliance cases. Always return valid JSON array format only, no markdown or additional text. Focus on real, documented cases and enforcement actions."))
            
            # Append user message with prompt
            chat.append(user(prompt))
            
            # Get response
            response = chat.sample()
            
            logger.info(f"Grok API call successful")
            
            # Extract content from response - it's a protobuf object
            # The response is in text format with structure:
            #   outputs {
            #     message {
            #       content: "JSON_CONTENT_HERE"
            #     }
            #   }
            response_str = str(response)
            
            # Use regex to extract the content field
            match = re.search(r'content:\s*"(.*?)"\s*role:', response_str, re.DOTALL)
            if match:
                content = match.group(1)
                # Unescape the JSON string (protobuf escaping format)
                content = content.replace('\\"', '"')      # \"  -> "
                content = content.replace('\\/', '/')      # \/  -> /
                content = content.replace("\\'", "'")      # \'  -> '
                content = content.replace('\\n', '\n')     # \n  -> newline
                content = content.replace('\\t', '\t')     # \t  -> tab
                content = content.replace('\\\\', '\\')    # \\ -> \ (do this last!)
                logger.debug(f"Extracted content via regex: {len(content)} characters")
            else:
                logger.warning("Could not extract content with regex, using full response string")
                content = response_str
            
            logger.info(f"Received {len(content)} characters of content")
            logger.debug(f"Response content (first 500 chars): {content[:500]}")
            
            # Parse JSON cases from content
            try:
                cases = json.loads(content)
                if not isinstance(cases, list):
                    logger.warning("Response is not a JSON array, wrapping in list")
                    cases = [cases]
                
                # Add metadata to each case
                for case in cases:
                    if "error" not in case:
                        # Store the xai response metadata
                        try:
                            usage_info = response.usage if hasattr(response, 'usage') else {}
                            meta_data = {
                                'model': self.model,
                                'api_provider': 'xai'
                            }
                            
                            if hasattr(usage_info, 'input_tokens'):
                                if 'usage' not in meta_data:
                                    meta_data['usage'] = {}
                                meta_data['usage']['input_tokens'] = usage_info.input_tokens
                            
                            if hasattr(usage_info, 'output_tokens'):
                                if 'usage' not in meta_data:
                                    meta_data['usage'] = {}
                                meta_data['usage']['output_tokens'] = usage_info.output_tokens
                            
                            if hasattr(response, 'citations'):
                                meta_data['citations'] = str(response.citations)
                            
                            if hasattr(response, 'server_side_tool_usage'):
                                meta_data['server_side_tool_usage'] = str(response.server_side_tool_usage)
                            
                            case['meta'] = meta_data
                        except Exception as e:
                            logger.warning(f"Could not extract all metadata: {e}")
                            case['meta'] = {
                                'model': self.model,
                                'api_provider': 'xai'
                            }
                
                logger.info(f"Successfully parsed {len(cases)} cases")
                return cases
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                logger.warning(f"Response content (first 500 chars): {content[:500]}")
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        try:
                            json_str = content[start:end].strip()
                            cases = json.loads(json_str)
                            if not isinstance(cases, list):
                                cases = [cases]
                            # Add metadata
                            for case in cases:
                                if "error" not in case:
                                    case['meta'] = {
                                        'model': self.model,
                                        'api_provider': 'xai'
                                    }
                            logger.info(f"Successfully parsed {len(cases)} cases from markdown")
                            return cases
                        except json.JSONDecodeError:
                            pass
                
                # Try to extract JSON array directly
                if "[" in content:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if end > start:
                        try:
                            json_str = content[start:end]
                            cases = json.loads(json_str)
                            if isinstance(cases, list):
                                # Add metadata
                                for case in cases:
                                    if "error" not in case:
                                        case['meta'] = {
                                            'model': self.model,
                                            'api_provider': 'xai'
                                        }
                                logger.info(f"Successfully parsed {len(cases)} cases from extracted JSON")
                                return cases
                        except json.JSONDecodeError:
                            pass
                
                logger.warning("Failed to parse any JSON from response")
                logger.warning(f"Full response content:\n{content}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self._call_grok(prompt, attempt + 1, max_retries, retry_delay)
                
                return [{"error": "Failed to parse JSON from content", "raw_sample": content[:200]}]
                
        except Exception as e:
            logger.warning(f"Grok API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                return self._call_grok(prompt, attempt + 1, max_retries, retry_delay)
            return [{"error": str(e), "status": "grok_api_error"}]
    
    def _call_deepseek(self, prompt: str, attempt: int, max_retries: int, retry_delay: int) -> List[Dict[str, Any]]:
        """Call DeepSeek API using requests"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a legal compliance analyst specializing in finding and analyzing real compliance cases. Always return valid JSON array format only, no markdown or additional text. Focus on real, documented cases and enforcement actions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 3000
            }
            
            response = requests.post(
                self.base_url, 
                json=data, 
                headers=headers, 
                timeout=120,
                stream=False
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if "choices" not in response_json or len(response_json["choices"]) == 0:
                logger.warning(f"Invalid API response format: {str(response_json)[:200]}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self._call_deepseek(prompt, attempt + 1, max_retries, retry_delay)
                return [{"error": "Invalid API response format"}]
            
            content = response_json["choices"][0]["message"]["content"]
            logger.info("DeepSeek API call successful, parsing response...")
            
            try:
                cases = json.loads(content)
                if not isinstance(cases, list):
                    logger.warning("Response is not a JSON array, wrapping in list")
                    cases = [cases]
                
                # Add metadata from DeepSeek response
                for case in cases:
                    if "error" not in case:
                        case['meta'] = {
                            'model': response_json.get('model'),
                            'usage': {
                                'input_tokens': response_json.get('usage', {}).get('prompt_tokens'),
                                'output_tokens': response_json.get('usage', {}).get('completion_tokens'),
                                'total_tokens': response_json.get('usage', {}).get('total_tokens'),
                            },
                            'finish_reason': response_json['choices'][0].get('finish_reason'),
                            'created': response_json.get('created'),
                            'api_provider': 'deepseek'
                        }
                
                logger.info(f"Successfully parsed {len(cases)} cases")
                return cases
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        try:
                            json_str = content[start:end].strip()
                            cases = json.loads(json_str)
                            if not isinstance(cases, list):
                                cases = [cases]
                            # Add metadata
                            for case in cases:
                                if "error" not in case:
                                    case['meta'] = {
                                        'model': response_json.get('model'),
                                        'usage': response_json.get('usage'),
                                        'api_provider': 'deepseek'
                                    }
                            logger.info(f"Successfully parsed {len(cases)} cases from markdown")
                            return cases
                        except json.JSONDecodeError:
                            pass
                
                logger.warning("Failed to parse response as JSON")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self._call_deepseek(prompt, attempt + 1, max_retries, retry_delay)
                
                return [{"error": "Failed to parse JSON response", "raw_response": content[:300]}]
                
        except requests.exceptions.Timeout as e:
            logger.warning(f"DeepSeek request timeout (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                return self._call_deepseek(prompt, attempt + 1, max_retries, retry_delay)
            return [{"error": f"API timeout after {max_retries} attempts"}]
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"DeepSeek request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
                return self._call_deepseek(prompt, attempt + 1, max_retries, retry_delay)
            return [{"error": str(e)}]


class CaseResearchAnalyzer:
    """Analyzer for researching compliance cases and building case database"""
    
    def __init__(self, api_key: str = None, provider: str = "deepseek", max_workers: int = 3, output_file: str = None):
        """Initialize with LLM API key
        
        Args:
            api_key: LLM API key (DeepSeek or XAI/Grok)
            provider: LLM provider ("deepseek" or "grok")
            max_workers: Number of parallel workers for processing multiple rules
            output_file: File to save accumulated cases (default: cases_database_TIMESTAMP.json)
        """
        if provider not in ["deepseek", "grok"]:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: deepseek, grok")
        
        if api_key is None:
            if provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY", "")
            elif provider == "grok":
                api_key = os.getenv("XAI_API_KEY", "")
        
        if not api_key:
            raise ValueError(f"{provider.upper()}_API_KEY environment variable not set")
        
        self.llm = LLMProvider(api_key, provider)
        self.max_workers = max_workers
        self.all_cases = []  # Accumulated cases
        
        # Set output file
        _base_name_file_path = os.path.basename(args.rule_file_path)
        _base_name_file_path = _base_name_file_path.split(".")[0]
        if output_file is None:
            output_file = f"{args.output_file_path}/{_base_name_file_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.output_file = output_file
        logger.info(f"Cases will be saved to: {self.output_file}")
    
    def research_rule(self, rule: str, query: str = "") -> List[Dict[str, Any]]:
        """Research cases for a single compliance rule
        
        Args:
            rule: The compliance rule to research
            query: Additional query context
            
        Returns:
            List of case dictionaries found for this rule
        """
        logger.info(f"Researching cases for rule: {rule[:60]}...")
        
        cases = self.llm.search_and_analyze_cases(rule, query)
        
        # Add rule reference to each case
        for case in cases:
            if "error" not in case:
                case['source_rule'] = rule
                case['research_timestamp'] = datetime.now().isoformat()
        
        # Add to accumulated cases
        self.all_cases.extend(cases)
        
        # Save immediately after getting results
        self._save_to_file()
        
        logger.info(f"Found {len(cases)} cases for this rule")
        return cases
    
    def research_multiple_rules(self, rules: List[str], query: str = "") -> Dict[str, Any]:
        """Research cases for multiple rules in parallel
        
        Args:
            rules: List of compliance rules to research
            query: Additional query context
            
        Returns:
            Summary of all research
        """
        logger.info(f"Starting case research for {len(rules)} rules in parallel with {self.max_workers} workers...")
        
        summary = {
            "total_rules": len(rules),
            "rules_processed": 0,
            "total_cases_found": 0,
            "start_time": datetime.now().isoformat(),
            "rule_summaries": []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.research_rule, rule, query): rule
                for rule in rules
            }
            
            for future in as_completed(futures):
                rule = futures[future]
                try:
                    cases = future.result()
                    summary["rules_processed"] += 1
                    
                    # Count non-error cases
                    valid_cases = [c for c in cases if "error" not in c]
                    summary["total_cases_found"] += len(valid_cases)
                    
                    summary["rule_summaries"].append({
                        "rule": rule[:80] + "..." if len(rule) > 80 else rule,
                        "cases_found": len(valid_cases),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.info(f"Completed: {summary['rules_processed']}/{len(rules)} rules (Total cases: {summary['total_cases_found']})")
                    
                except Exception as e:
                    logger.error(f"Error researching rule '{rule[:50]}...': {e}")
                    summary["rule_summaries"].append({
                        "rule": rule[:80] + "..." if len(rule) > 80 else rule,
                        "error": str(e)
                    })
        
        summary["end_time"] = datetime.now().isoformat()
        return summary
    
    def _save_to_file(self) -> None:
        """Save accumulated cases to JSON file as a list"""
        try:
            output_dir = os.path.dirname(self.output_file) or "."
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as a simple list of cases
            # Each case already contains 'source_rule' and 'research_timestamp'
            output = self.all_cases
            
            # Write to file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Saved {len(self.all_cases)} cases to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
    
    def get_cases_summary(self) -> Dict[str, Any]:
        """Get summary of researched cases"""
        if not self.all_cases:
            return {"total_cases": 0, "message": "No cases researched yet"}
        
        # Count by compliance status
        compliant = sum(1 for c in self.all_cases if c.get("compliance_with_rule") == "COMPLIES")
        violates = sum(1 for c in self.all_cases if c.get("compliance_with_rule") == "VIOLATES")
        errors = sum(1 for c in self.all_cases if "error" in c)
        
        # Count by severity
        severity_count = {}
        for case in self.all_cases:
            if "error" not in case and "violation_severity" in case:
                severity = case["violation_severity"]
                severity_count[severity] = severity_count.get(severity, 0) + 1
        
        return {
            "total_cases": len(self.all_cases),
            "valid_cases": len(self.all_cases) - errors,
            "compliant_cases": compliant,
            "violating_cases": violates,
            "error_cases": errors,
            "violation_severity_distribution": severity_count,
            "output_file": self.output_file
        }


def main():
    

    """Main entry point for case research analyzer"""
    print("=" * 80)
    print("Compliance Case Research and Analysis System")
    print("=" * 80)
    
    # Auto-detect which API provider to use
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    grok_key = os.getenv('XAI_API_KEY', '')
    
    # Try to load from config if env vars not set
    if not deepseek_key and not grok_key:
        try:
            from config import ds_api_key
            if ds_api_key:
                deepseek_key = ds_api_key
        except:
            pass
        
        try:
            from config import xai_key
            if xai_key:
                grok_key = xai_key
        except:
            pass
    
    if grok_key:
        provider = "grok"
        api_key = grok_key
        print("\n✓ Using Grok API (via xai_sdk)")
    elif deepseek_key:
        provider = "deepseek"
        api_key = deepseek_key
        print("\n✓ Using DeepSeek API")
    else:
        print("\n✗ No API key found")
        print("Please set one of:")
        print("  export DEEPSEEK_API_KEY='sk-xxxxx'")
        print("  export XAI_API_KEY='your-xai-key'")
        return
    
    # Initialize analyzer with 3 parallel workers (to manage API rate limits)
    analyzer = CaseResearchAnalyzer(api_key, provider=provider, max_workers=10)
    
    # Example compliance rules to research
    # with open('rules.json', 'r') as file:
    with open(args.rule_file_path, 'r') as file:
        rules = json.load(file)
    



    
    # rules = [
    #     "Data Protection: All personal data must be encrypted at rest and in transit according to industry standards",
    #     "Privacy Compliance: Users must provide explicit consent before any data collection or processing",
    #     "Access Control: Only authorized personnel with appropriate clearance can access sensitive information"
    # ]
    
    print(f"\n🔍 Starting case research for {len(rules)} compliance rules...")
    print("(Each rule will trigger ONE LLM API call to find relevant cases)")
    print("(Cases will be saved to file after EACH rule is processed)")
    print()
    
    # Research rules in parallel
    summary = analyzer.research_multiple_rules(rules)
    
    # Display summary
    print("\n" + "=" * 80)
    print("📊 Case Research Summary")
    print("=" * 80)
    print(f"Rules processed: {summary['rules_processed']}/{summary['total_rules']}")
    print(f"Total cases found: {summary['total_cases_found']}")
    print(f"Start time: {summary['start_time']}")
    print(f"End time: {summary['end_time']}")
    
    # Display per-rule results
    print("\nPer-Rule Summary:")
    for rule_summary in summary['rule_summaries']:
        if 'error' in rule_summary:
            print(f"  ✗ {rule_summary['rule']}")
            print(f"    Error: {rule_summary['error']}")
        else:
            print(f"  ✓ {rule_summary['rule']}")
            print(f"    Cases found: {rule_summary['cases_found']}")
    
    # Display overall case statistics
    cases_summary = analyzer.get_cases_summary()
    print("\n" + "=" * 80)
    print("📈 Cases Database Statistics")
    print("=" * 80)
    print(f"Total cases in database: {cases_summary['total_cases']}")
    print(f"Valid cases: {cases_summary['valid_cases']}")
    print(f"Cases showing compliance: {cases_summary['compliant_cases']}")
    print(f"Cases showing violations: {cases_summary['violating_cases']}")
    if cases_summary['violation_severity_distribution']:
        print(f"Violation severity distribution:")
        for severity, count in cases_summary['violation_severity_distribution'].items():
            print(f"  - {severity}: {count} cases")
    
    print("\n" + "=" * 80)
    print(f"💾 Cases database saved to: {cases_summary['output_file']}")
    print("=" * 80)
    
    # Display sample cases (if any)
    if analyzer.all_cases and len(analyzer.all_cases) > 0:
        print("\n📋 Sample Case Information (First 2 cases):")
        for i, case in enumerate(analyzer.all_cases[:2], 1):
            if "error" not in case:
                print(f"\n  Case {i}: {case.get('case_name', 'Unknown')}")
                print(f"    - Defendant: {case.get('defendant', 'Unknown')}")
                print(f"    - Compliance: {case.get('compliance_with_rule', 'Unknown')}")
                print(f"    - Severity: {case.get('violation_severity', 'N/A')}")
                print(f"    - Reference: {case.get('reference_link', 'N/A')}")
                if 'meta' in case:
                    print(f"    - API Provider: {case['meta'].get('api_provider')}")
                    print(f"    - Model: {case['meta'].get('model')}")


if __name__ == "__main__":
    main()
