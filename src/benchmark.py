#!/usr/bin/env python3
"""
LLM Benchmark Tool

A tool for parallel LLM prompt benchmarking across multiple models via OpenRouter API.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_config(config_path: str = "config/models.json") -> dict:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def query_model(model_id: str, prompt: str, api_key: str, temperature: float = 0.7, max_tokens: int = 1000) -> dict:
    """Send a prompt to a single model via OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/clawson1717/llm-benchmark-tool",
        "X-Title": "LLM Benchmark Tool"
    }
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        elapsed_time = time.time() - start_time
        data = response.json()
        
        # Extract response content
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        
        return {
            "success": True,
            "model": model_id,
            "response": content,
            "response_time_seconds": round(elapsed_time, 2),
            "tokens_used": usage,
            "timestamp": datetime.now().isoformat()
        }
    
    except requests.exceptions.RequestException as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "model": model_id,
            "error": str(e),
            "response_time_seconds": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat()
        }


def benchmark_models(models: list, prompt: str, api_key: str, max_workers: int = 5) -> dict:
    """Run benchmarks on multiple models in parallel."""
    results = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "models_tested": len(models),
        "responses": []
    }
    
    print(f"\n🚀 Benchmarking {len(models)} models with prompt:")
    print(f'   "{prompt[:80]}{"..." if len(prompt) > 80 else ""}"\n')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(query_model, model["id"], prompt, api_key, model.get("temperature", 0.7), model.get("max_tokens", 1000)): model
            for model in models
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results["responses"].append(result)
                
                status = "✅" if result["success"] else "❌"
                print(f"{status} {model['name']}: {result['response_time_seconds']:.2f}s")
                
            except Exception as e:
                print(f"❌ {model['name']}: Error - {e}")
                results["responses"].append({
                    "success": False,
                    "model": model["id"],
                    "name": model["name"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
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


def print_summary(results: dict):
    """Print a summary of benchmark results."""
    print("\n" + "="*60)
    print("📊 BENCHMARK SUMMARY")
    print("="*60)
    
    successful = [r for r in results["responses"] if r.get("success")]
    failed = [r for r in results["responses"] if not r.get("success")]
    
    print(f"\nTotal models tested: {len(results['responses'])}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n⏱️  Response Times (sorted):")
        sorted_responses = sorted(successful, key=lambda x: x["response_time_seconds"])
        for r in sorted_responses:
            model_name = r["model"].split("/")[-1]
            print(f"   {model_name:30s} {r['response_time_seconds']:6.2f}s")
    
    if failed:
        print("\n❌ Failures:")
        for r in failed:
            print(f"   {r['model']}: {r.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple LLMs via OpenRouter API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/benchmark.py "What is the capital of France?"
  python src/benchmark.py -c config/models.json -p "Explain recursion"
  python src/benchmark.py --prompt-file examples/prompts.txt
        """
    )
    
    parser.add_argument("prompt", nargs="?", help="The prompt to send to models")
    parser.add_argument("-c", "--config", default="config/models.json", 
                        help="Path to models config file (default: config/models.json)")
    parser.add_argument("-p", "--prompt-file", 
                        help="Path to file containing the prompt")
    parser.add_argument("-o", "--output", default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument("-w", "--workers", type=int, default=5,
                        help="Number of parallel workers (default: 5)")
    parser.add_argument("--show-responses", action="store_true",
                        help="Display model responses in output")
    
    args = parser.parse_args()
    
    # Get prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        print("\nError: Must provide either a prompt argument or --prompt-file")
        sys.exit(1)
    
    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key at: https://openrouter.ai/keys")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config)
        models = config.get("models", [])
        if not models:
            print("Error: No models defined in configuration")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in config file: {args.config}")
        sys.exit(1)
    
    # Run benchmark
    results = benchmark_models(models, prompt, api_key, args.workers)
    
    # Save results
    output_file = save_results(results, args.output)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print_summary(results)
    
    # Optionally show responses
    if args.show_responses:
        print("\n" + "="*60)
        print("📝 MODEL RESPONSES")
        print("="*60)
        for r in results["responses"]:
            if r.get("success"):
                model_name = r["model"].split("/")[-1]
                print(f"\n{'─'*60}")
                print(f"🤖 {model_name}")
                print(f"{'─'*60}")
                print(r["response"])


if __name__ == "__main__":
    main()
