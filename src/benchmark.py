#!/usr/bin/env python3
"""
LLM Benchmark Tool

A tool for parallel LLM prompt benchmarking across multiple models via OpenRouter.
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Default API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_config(config_path: str = "config/models.json") -> dict:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def is_transient_error(exception) -> bool:
    """Check if an exception represents a transient error that should be retried.
    
    Transient errors include: timeouts, connection errors, 5xx server errors.
    Non-transient (client) errors like 4xx are not retried.
    """
    if isinstance(exception, requests.exceptions.Timeout):
        return True
    if isinstance(exception, requests.exceptions.ConnectionError):
        return True
    if isinstance(exception, requests.exceptions.HTTPError):
        # Check if it's a 5xx error (server error) - these should be retried
        response = exception.response
        if response.status_code >= 500:
            return True
        # 4xx errors are client errors - don't retry (invalid API key, bad request, etc.)
        return False
    return False


def query_model(model_id: str, prompt: str, api_key: str, max_retries: int = 3) -> dict:
    """Send a single prompt to a model via OpenRouter API with retry logic.
    
    Args:
        model_id: The model identifier (e.g., "openai/gpt-4o")
        prompt: The prompt text to send
        api_key: The OpenRouter API key
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        Dictionary containing the response data or error information
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/clawson1717/llm-benchmark-tool",
        "X-Title": "LLM Benchmark Tool"
    }
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    start_time = time.time()
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            elapsed_time = time.time() - start_time
            
            return {
                "model": model_id,
                "success": True,
                "response": data["choices"][0]["message"]["content"],
                "response_time": round(elapsed_time, 2),
                "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                "error": None
            }
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            last_exception = e
            
            # Check if we should retry this error
            if attempt < max_retries and is_transient_error(e):
                # Exponential backoff: 1s, 2s, 4s between retries
                delay = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {model_id}: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                # Don't retry on non-transient errors or if we've exhausted retries
                if not is_transient_error(e):
                    logger.info(f"Request failed for {model_id} with non-transient error (no retry): {str(e)}")
                break
    
    # All retries exhausted or non-retryable error
    elapsed_time = time.time() - start_time
    error_msg = str(last_exception)
    if isinstance(last_exception, (KeyError, IndexError)):
        error_msg = f"Parse error: {error_msg}"
    
    return {
        "model": model_id,
        "success": False,
        "response": None,
        "response_time": round(elapsed_time, 2),
        "tokens_used": 0,
        "error": error_msg
    }


def run_benchmark(prompt: str, models: list, api_key: str, max_workers: int = 5, max_retries: int = 3) -> dict:
    """Run benchmark across multiple models in parallel.
    
    Args:
        prompt: The prompt to send to all models
        models: List of model configuration dictionaries
        api_key: The OpenRouter API key
        max_workers: Maximum number of concurrent requests
        max_retries: Maximum retry attempts per model on transient errors
    
    Returns:
        Dictionary containing all benchmark results
    """
    results = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "model_count": len(models),
        "max_retries": max_retries,
        "results": []
    }
    
    # Create progress bar
    pbar = tqdm(
        total=len(models),
        desc="Starting benchmark",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ncols=80,
        unit="model"
    )
    
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(query_model, model["id"], prompt, api_key, max_retries): model
            for model in models
        }
        
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results["results"].append(result)
                completed += 1
                
                # Update progress bar with current status
                status_icon = "✓" if result["success"] else "✗"
                pbar.set_description(
                    f"{status_icon} {model['name'][:30]:<30} ({result['response_time']:.2f}s)"
                )
                pbar.update(1)
            except Exception as e:
                completed += 1
                pbar.set_description(f"✗ {model['name'][:30]:<30} (Error)")
                results["results"].append({
                    "model": model["id"],
                    "success": False,
                    "response": None,
                    "response_time": 0,
                    "tokens_used": 0,
                    "error": str(e)
                })
                pbar.update(1)
    
    pbar.close()
    
    # Sort results by model name for consistency
    results["results"].sort(key=lambda x: x["model"])
    
    # Print summary of completed queries
    print(f"\n✓ Completed {completed}/{len(models)} model queries")
    
    return results


def save_results(results: dict, output_dir: str = "results") -> str:
    """Save benchmark results to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return str(filepath)


def save_results_csv(results: dict, output_dir: str = "results") -> str:
    """Save benchmark results to a CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}.csv"
    filepath = output_path / filename
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow(['model', 'status', 'response_time', 'tokens_used', 'response_preview'])
        # Write data rows
        for result in results['results']:
            status = 'success' if result['success'] else 'failed'
            response_preview = ''
            if result['success'] and result['response']:
                response_preview = result['response'][:200].replace('\n', ' ').replace('\r', '')
            elif not result['success'] and result['error']:
                response_preview = result['error'][:200].replace('\n', ' ').replace('\r', '')
            
            writer.writerow([
                result['model'],
                status,
                result['response_time'],
                result.get('tokens_used', 0),
                response_preview
            ])
    
    return str(filepath)


def print_summary(results: dict):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Prompt: {results['prompt'][:80]}...")
    print(f"Models tested: {results['model_count']}")
    print(f"Timestamp: {results['timestamp']}")
    print("-" * 60)
    
    for result in results["results"]:
        status = "✓" if result["success"] else "✗"
        print(f"\n{status} {result['model']}")
        print(f"  Response time: {result['response_time']:.2f}s")
        if result["success"]:
            print(f"  Tokens used: {result['tokens_used']}")
            preview = result['response'][:150].replace('\n', ' ')
            print(f"  Preview: {preview}...")
        else:
            print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple LLMs via OpenRouter API"
    )
    parser.add_argument(
        "prompt",
        help="The prompt to send to all models"
    )
    parser.add_argument(
        "--config",
        default="config/models.json",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum parallel requests"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed API requests (default: 3)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config)
        models = config.get("models", [])
        if not models:
            print("Error: No models found in configuration")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration: {e}")
        sys.exit(1)
    
    print(f"\nRunning benchmark with {len(models)} models...")
    print(f"Prompt: {args.prompt}")
    print(f"Max retries per request: {args.max_retries}\n")
    
    # Run benchmark
    results = run_benchmark(args.prompt, models, args.api_key, args.max_workers, args.max_retries)
    
    # Save results based on format
    if args.format == "csv":
        output_file = save_results_csv(results, args.output_dir)
    else:
        output_file = save_results(results, args.output_dir)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
