#!/usr/bin/env python3
"""
LLM Benchmark Results Viewer

A tool for generating formatted reports from benchmark JSON output.
Supports markdown, HTML, and console output formats.
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from html import escape


def load_results(filepath: str) -> dict:
    """Load benchmark results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_results_csv(filepath: str) -> dict:
    """Load benchmark results from a CSV file."""
    results = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {
                'model': row['model'],
                'success': row['status'] == 'success',
                'response_time': float(row['response_time']),
                'tokens_used': int(row['tokens_used']) if row['tokens_used'] else 0,
                'response': row['response_preview'] if row['status'] == 'success' else None,
                'error': row['response_preview'] if row['status'] != 'success' else None
            }
            results.append(result)
    
    # Extract timestamp from filename if possible
    filename = Path(filepath).name
    timestamp = "Unknown"
    if filename.startswith('benchmark_') and len(filename) >= 22:
        try:
            ts_str = filename[10:25]  # Extract YYYYMMDD_HHMMSS
            timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").isoformat()
        except ValueError:
            pass
    
    return {
        'prompt': 'Loaded from CSV (prompt not available)',
        'timestamp': timestamp,
        'model_count': len(results),
        'results': results
    }


def calculate_statistics(results: List[dict]) -> dict:
    """Calculate summary statistics from benchmark results."""
    total_models = len(results)
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    success_rate = (len(successful) / total_models * 100) if total_models > 0 else 0
    
    if successful:
        response_times = [r["response_time"] for r in successful]
        avg_response_time = sum(response_times) / len(response_times)
        fastest = min(successful, key=lambda x: x["response_time"])
        slowest = max(successful, key=lambda x: x["response_time"])
        total_tokens = sum(r["tokens_used"] for r in successful)
    else:
        avg_response_time = 0
        fastest = None
        slowest = None
        total_tokens = 0
    
    return {
        "total_models": total_models,
        "successful_count": len(successful),
        "failed_count": len(failed),
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "fastest": fastest,
        "slowest": slowest,
        "total_tokens": total_tokens
    }


def format_markdown_table(results: List[dict]) -> str:
    """Generate a markdown table of model responses."""
    lines = [
        "| Model | Status | Response Time | Tokens | Preview |",
        "|-------|--------|---------------|--------|---------|"
    ]
    
    for result in results:
        status = "✅ Success" if result["success"] else "❌ Failed"
        time_str = f"{result['response_time']:.2f}s"
        tokens = str(result.get("tokens_used", 0)) if result["success"] else "N/A"
        
        if result["success"] and result["response"]:
            preview = result["response"][:80].replace("\n", " ").replace("|", "\\|")
            if len(result["response"]) > 80:
                preview += "..."
        else:
            preview = result.get("error", "No response")[:60].replace("\n", " ").replace("|", "\\|")
        
        lines.append(f"| {result['model']} | {status} | {time_str} | {tokens} | {preview} |")
    
    return "\n".join(lines)


def format_side_by_side(results: List[dict]) -> str:
    """Generate a side-by-side comparison view of responses."""
    successful = [r for r in results if r["success"]]
    
    if not successful:
        return "*No successful responses to compare.*"
    
    lines = ["## Side-by-Side Response Comparison\n"]
    
    for result in successful:
        lines.append(f"### {result['model']}")
        lines.append(f"**Response Time:** {result['response_time']:.2f}s | **Tokens:** {result.get('tokens_used', 'N/A')}\n")
        lines.append("```")
        lines.append(result.get("response", "No response"))
        lines.append("```\n")
    
    return "\n".join(lines)


def generate_markdown(data: dict, include_side_by_side: bool = False) -> str:
    """Generate a complete markdown report from benchmark data."""
    prompt = data.get("prompt", "N/A")
    timestamp = data.get("timestamp", "N/A")
    results = data.get("results", [])
    
    stats = calculate_statistics(results)
    
    lines = [
        "# LLM Benchmark Results\n",
        "## Benchmark Overview\n",
        f"**Timestamp:** {timestamp}\n",
        f"**Models Tested:** {stats['total_models']}\n",
        "### Prompt Used",
        "```",
        prompt,
        "```\n",
        "## Summary Statistics\n",
        f"- **Success Rate:** {stats['success_rate']:.1f}% ({stats['successful_count']}/{stats['total_models']})",
        f"- **Average Response Time:** {stats['avg_response_time']:.2f}s",
        f"- **Total Tokens Used:** {stats['total_tokens']:,}",
    ]
    
    if stats["fastest"]:
        lines.append(f"- **Fastest Model:** {stats['fastest']['model']} ({stats['fastest']['response_time']:.2f}s)")
    
    if stats["slowest"]:
        lines.append(f"- **Slowest Model:** {stats['slowest']['model']} ({stats['slowest']['response_time']:.2f}s)")
    
    lines.append("\n## Results Table\n")
    lines.append(format_markdown_table(results))
    
    if include_side_by_side:
        lines.append("\n")
        lines.append(format_side_by_side(results))
    
    if stats["failed_count"] > 0:
        lines.append("\n## Failed Requests\n")
        for result in results:
            if not result["success"]:
                lines.append(f"- **{result['model']}:** {result.get('error', 'Unknown error')}")
    
    return "\n".join(lines)


def generate_html(data: dict, include_side_by_side: bool = False) -> str:
    """Generate an HTML report from benchmark data."""
    prompt = escape(data.get("prompt", "N/A"))
    timestamp = escape(str(data.get("timestamp", "N/A")))
    results = data.get("results", [])
    
    stats = calculate_statistics(results)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Benchmark Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #555; }}
        .prompt-box {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .status-success {{ color: #27ae60; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
        .response-preview {{
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .comparison-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .response-card {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .response-header {{
            background: #f8f9fa;
            padding: 10px;
            margin: -20px -20px 15px -20px;
            border-radius: 8px 8px 0 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .response-content {{
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .error-section {{
            background: #fdf2f2;
            border: 1px solid #fca5a5;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }}
        .error-item {{ color: #c53030; margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>🚀 LLM Benchmark Results</h1>
    
    <h2>Benchmark Overview</h2>
    <p><strong>Timestamp:</strong> {timestamp}</p>
    <p><strong>Models Tested:</strong> {stats['total_models']}</p>
    
    <h3>Prompt Used</h3>
    <div class="prompt-box">{prompt}</div>
    
    <h2>Summary Statistics</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Success Rate</div>
            <div class="stat-value">{stats['success_rate']:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Successful / Total</div>
            <div class="stat-value">{stats['successful_count']} / {stats['total_models']}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Response Time</div>
            <div class="stat-value">{stats['avg_response_time']:.2f}s</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Tokens Used</div>
            <div class="stat-value">{stats['total_tokens']:,}</div>
        </div>
"""
    
    if stats["fastest"]:
        html += f"""
        <div class="stat-card">
            <div class="stat-label">Fastest Model</div>
            <div class="stat-value">{escape(stats['fastest']['model'])}</div>
            <div>{stats['fastest']['response_time']:.2f}s</div>
        </div>
"""
    
    html += """
    </div>
    
    <h2>Results Table</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Response Time</th>
                <th>Tokens</th>
                <th>Preview</th>
            </tr>
        </thead>
        <tbody>
"""
    
    for result in results:
        status_class = "status-success" if result["success"] else "status-failed"
        status_text = "✓ Success" if result["success"] else "✗ Failed"
        time_str = f"{result['response_time']:.2f}s"
        tokens = str(result.get("tokens_used", "N/A")) if result["success"] else "N/A"
        
        if result["success"] and result["response"]:
            preview = escape(result["response"][:80])
        else:
            preview = escape(result.get("error", "No response")[:60])
        
        html += f"""
            <tr>
                <td>{escape(result['model'])}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{time_str}</td>
                <td>{tokens}</td>
                <td class="response-preview">{preview}</td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
"""
    
    if include_side_by_side and stats["successful_count"] > 0:
        html += """
    <h2>Side-by-Side Response Comparison</h2>
    <div class="comparison-section">
"""
        for result in results:
            if result["success"]:
                response_text = escape(result.get("response", "No response"))
                html += f"""
        <div class="response-card">
            <div class="response-header">
                <strong>{escape(result['model'])}</strong>
                <span style="float: right;">{result['response_time']:.2f}s | {result.get('tokens_used', 'N/A')} tokens</span>
            </div>
            <div class="response-content">{response_text}</div>
        </div>
"""
        html += """
    </div>
"""
    
    if stats["failed_count"] > 0:
        html += """
    <div class="error-section">
        <h2>Failed Requests</h2>
"""
        for result in results:
            if not result["success"]:
                error_msg = escape(result.get("error", "Unknown error"))
                html += f"""
        <div class="error-item">
            <strong>{escape(result['model'])}:</strong> {error_msg}
        </div>
"""
        html += """
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    return html


def generate_console(data: dict, include_side_by_side: bool = False) -> str:
    """Generate a pretty-printed console report from benchmark data."""
    prompt = data.get("prompt", "N/A")
    timestamp = data.get("timestamp", "N/A")
    results = data.get("results", [])
    
    stats = calculate_statistics(results)
    
    lines = [
        "=" * 70,
        "  LLM BENCHMARK RESULTS",
        "=" * 70,
        "",
        f"  Timestamp: {timestamp}",
        f"  Models Tested: {stats['total_models']}",
        "",
        "  PROMPT:",
        "  " + "-" * 66,
    ]
    
    # Wrap prompt lines
    prompt_lines = prompt.split("\n")
    for pl in prompt_lines:
        while pl:
            chunk = pl[:66]
            pl = pl[66:]
            lines.append(f"  {chunk}")
    
    lines.extend([
        "  " + "-" * 66,
        "",
        "  SUMMARY STATISTICS:",
        f"    Success Rate:      {stats['success_rate']:.1f}% ({stats['successful_count']}/{stats['total_models']})",
        f"    Avg Response Time: {stats['avg_response_time']:.2f}s",
        f"    Total Tokens Used: {stats['total_tokens']:,}",
    ])
    
    if stats["fastest"]:
        lines.append(f"    Fastest Model:     {stats['fastest']['model']} ({stats['fastest']['response_time']:.2f}s)")
    
    if stats["slowest"]:
        lines.append(f"    Slowest Model:     {stats['slowest']['model']} ({stats['slowest']['response_time']:.2f}s)")
    
    lines.extend([
        "",
        "  RESULTS TABLE:",
        "  " + "-" * 68,
        f"  {'Model':<35} {'Status':<12} {'Time':<10} {'Tokens':<10}",
        "  " + "-" * 68,
    ])
    
    for result in results:
        status = "✓ Success" if result["success"] else "✗ Failed"
        time_str = f"{result['response_time']:.2f}s"
        tokens = str(result.get("tokens_used", "N/A")) if result["success"] else "N/A"
        model_name = result['model'][:34]
        lines.append(f"  {model_name:<35} {status:<12} {time_str:<10} {tokens:<10}")
    
    lines.append("  " + "-" * 68)
    
    if include_side_by_side:
        lines.extend([
            "",
            "  SIDE-BY-SIDE COMPARISON:",
            "  " + "=" * 68,
        ])
        
        for result in results:
            if result["success"]:
                lines.extend([
                    "",
                    f"  MODEL: {result['model']}",
                    f"  Time: {result['response_time']:.2f}s | Tokens: {result.get('tokens_used', 'N/A')}",
                    "  " + "-" * 68,
                ])
                response_lines = result.get("response", "No response").split("\n")
                for rl in response_lines[:20]:  # Limit to 20 lines per response
                    wrapped = rl[:66]
                    lines.append(f"  {wrapped}")
                if len(response_lines) > 20:
                    lines.append("  ... (truncated)")
                lines.append("  " + "-" * 68)
    
    if stats["failed_count"] > 0:
        lines.extend([
            "",
            "  FAILED REQUESTS:",
            "  " + "-" * 68,
        ])
        for result in results:
            if not result["success"]:
                error = result.get("error", "Unknown error")[:60]
                lines.append(f"    {result['model']}: {error}")
    
    lines.extend([
        "",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# =============================================================================
# HISTORICAL COMPARISON FEATURE
# =============================================================================

def load_multiple_results(filepaths: List[str]) -> List[dict]:
    """Load multiple benchmark result files."""
    results = []
    for filepath in filepaths:
        try:
            if filepath.lower().endswith('.csv'):
                data = load_results_csv(filepath)
            else:
                data = load_results(filepath)
            data['source_file'] = filepath
            results.append(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {filepath}: {e}")
    return results


def extract_run_label(filepath: str, index: int) -> str:
    """Extract a short label for a run from its filepath."""
    filename = Path(filepath).name
    # Try to extract date from benchmark_YYYYMMDD_HHMMSS.json
    if filename.startswith('benchmark_') and len(filename) >= 22:
        try:
            date_str = filename[10:18]  # YYYYMMDD
            time_str = filename[19:25]  # HHMMSS
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except:
            pass
    return f"Run {index + 1}"


def build_comparison_data(results_list: List[dict]) -> Tuple[Dict[str, List[dict]], List[str]]:
    """
    Build comparison data structure.
    Returns: (model_data dict mapping model names to list of results per run, run_labels list)
    """
    run_labels = []
    all_models = set()
    
    # Collect all models and run labels
    for i, data in enumerate(results_list):
        run_labels.append(extract_run_label(data.get('source_file', ''), i))
        for result in data.get('results', []):
            all_models.add(result['model'])
    
    # Build model -> list of results (one per run) mapping
    model_data = {model: [] for model in all_models}
    
    for data in results_list:
        run_results = {r['model']: r for r in data.get('results', [])}
        for model in all_models:
            if model in run_results:
                model_data[model].append(run_results[model])
            else:
                # Model not present in this run
                model_data[model].append(None)
    
    return model_data, run_labels


def calculate_trend(current: float, previous: float) -> Tuple[str, str, float]:
    """
    Calculate trend indicator and percentage change.
    Returns: (arrow, direction_text, percentage)
    """
    if previous == 0:
        return ("→", "same", 0.0)
    
    change = current - previous
    pct_change = (change / previous) * 100
    
    if change < 0:
        return ("↓", "faster", abs(pct_change))  # Lower time is better
    elif change > 0:
        return ("↑", "slower", abs(pct_change))  # Higher time is worse
    else:
        return ("→", "same", 0.0)


def calculate_success_trend(current: bool, previous: bool) -> Tuple[str, str]:
    """Calculate success rate trend indicator."""
    if current and not previous:
        return ("↑", "improved")
    elif not current and previous:
        return ("↓", "degraded")
    elif current and previous:
        return ("→", "stable")
    else:
        return ("→", "still failing")


def format_comparison_markdown(model_data: Dict[str, List[dict]], run_labels: List[str]) -> str:
    """Generate a markdown comparison table."""
    lines = [
        "# LLM Benchmark Historical Comparison\n",
        "## Response Time Comparison\n",
    ]
    
    # Build header
    header = "| Model | Avg |"
    separator = "|-------|-----|"
    for label in run_labels:
        header += f" {label} |"
        separator += "---------|"
    
    # Add trend columns between runs
    if len(run_labels) > 1:
        for i in range(len(run_labels) - 1):
            header += f" Trend{i+1} |"
            separator += "---------|"
    
    lines.append(header)
    lines.append(separator)
    
    # Build rows for each model
    for model in sorted(model_data.keys()):
        results = model_data[model]
        
        # Calculate average response time across successful runs
        successful_times = [r['response_time'] for r in results if r and r['success']]
        avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        row = f"| {model} | {avg_time:.2f}s |"
        
        # Add response times for each run
        for result in results:
            if result is None:
                row += " N/A |"
            elif result['success']:
                row += f" {result['response_time']:.2f}s |"
            else:
                row += " ❌ |"
        
        # Add trend indicators
        for i in range(len(results) - 1):
            curr = results[i]
            next_res = results[i + 1]
            
            if curr is None or next_res is None:
                row += " N/A |"
            elif not curr['success'] or not next_res['success']:
                # Show success status trend if either failed
                curr_success = curr['success'] if curr else False
                next_success = next_res['success'] if next_res else False
                arrow, status = calculate_success_trend(next_success, curr_success)
                row += f" {arrow} {status} |"
            else:
                arrow, direction, pct = calculate_trend(next_res['response_time'], curr['response_time'])
                row += f" {arrow} {pct:.1f}% |"
        
        lines.append(row)
    
    lines.append("\n### Legend")
    lines.append("- **↓** = Faster response time (improvement)")
    lines.append("- **↑** = Slower response time (regression)")
    lines.append("- **→** = No change")
    lines.append("- **Avg** = Average response time across all successful runs")
    
    # Add success rate comparison
    lines.append("\n## Success Rate Comparison\n")
    
    header = "| Model | Overall |"
    separator = "|-------|---------|"
    for label in run_labels:
        header += f" {label} |"
        separator += "-------|"
    
    lines.append(header)
    lines.append(separator)
    
    for model in sorted(model_data.keys()):
        results = model_data[model]
        successful = sum(1 for r in results if r and r['success'])
        overall_rate = (successful / len(results)) * 100 if results else 0
        
        row = f"| {model} | {overall_rate:.0f}% |"
        
        for result in results:
            if result is None:
                row += " N/A |"
            elif result['success']:
                row += " ✅ |"
            else:
                row += " ❌ |"
        
        lines.append(row)
    
    return "\n".join(lines)


def format_comparison_html(model_data: Dict[str, List[dict]], run_labels: List[str]) -> str:
    """Generate an HTML comparison report."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Benchmark Historical Comparison</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #3498db;
            color: white;
            font-weight: 600;
        }
        tr:hover { background: #f5f5f5; }
        .trend-up { color: #e74c3c; font-weight: bold; }
        .trend-down { color: #27ae60; font-weight: bold; }
        .trend-same { color: #7f8c8d; }
        .success { color: #27ae60; }
        .failed { color: #e74c3c; }
        .legend {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .avg-cell { font-weight: bold; background: #ecf0f1; }
    </style>
</head>
<body>
    <h1>📊 LLM Benchmark Historical Comparison</h1>
    
    <h2>Response Time Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Avg</th>
"""
    
    for label in run_labels:
        html += f"                <th>{label}</th>\n"
    
    if len(run_labels) > 1:
        for i in range(len(run_labels) - 1):
            html += f"                <th>Trend {i+1}</th>\n"
    
    html += """            </tr>
        </thead>
        <tbody>
"""
    
    for model in sorted(model_data.keys()):
        results = model_data[model]
        
        successful_times = [r['response_time'] for r in results if r and r['success']]
        avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        html += f"            <tr>\n"
        html += f"                <td><strong>{escape(model)}</strong></td>\n"
        html += f"                <td class='avg-cell'>{avg_time:.2f}s</td>\n"
        
        for result in results:
            if result is None:
                html += "                <td>N/A</td>\n"
            elif result['success']:
                html += f"                <td>{result['response_time']:.2f}s</td>\n"
            else:
                html += "                <td class='failed'>❌ Failed</td>\n"
        
        for i in range(len(results) - 1):
            curr = results[i]
            next_res = results[i + 1]
            
            if curr is None or next_res is None:
                html += "                <td>N/A</td>\n"
            elif not curr['success'] or not next_res['success']:
                curr_success = curr['success'] if curr else False
                next_success = next_res['success'] if next_res else False
                arrow, status = calculate_success_trend(next_success, curr_success)
                if status == "improved":
                    html += f"                <td class='trend-down'>{arrow} {status}</td>\n"
                elif status == "degraded":
                    html += f"                <td class='trend-up'>{arrow} {status}</td>\n"
                else:
                    html += f"                <td class='trend-same'>{arrow} {status}</td>\n"
            else:
                arrow, direction, pct = calculate_trend(next_res['response_time'], curr['response_time'])
                if direction == "faster":
                    html += f"                <td class='trend-down'>{arrow} {pct:.1f}%</td>\n"
                elif direction == "slower":
                    html += f"                <td class='trend-up'>{arrow} {pct:.1f}%</td>\n"
                else:
                    html += f"                <td class='trend-same'>{arrow} 0%</td>\n"
        
        html += "            </tr>\n"
    
    html += """        </tbody>
    </table>
    
    <div class="legend">
        <strong>Legend:</strong><br>
        <span class="trend-down">↓ Green</span> = Faster/better (improvement)<br>
        <span class="trend-up">↑ Red</span> = Slower/worse (regression)<br>
        <span class="trend-same">→ Gray</span> = No change<br>
        <strong>Avg</strong> = Average response time across all successful runs
    </div>
    
    <h2>Success Rate Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Overall</th>
"""
    
    for label in run_labels:
        html += f"                <th>{label}</th>\n"
    
    html += """            </tr>
        </thead>
        <tbody>
"""
    
    for model in sorted(model_data.keys()):
        results = model_data[model]
        successful = sum(1 for r in results if r and r['success'])
        overall_rate = (successful / len(results)) * 100 if results else 0
        
        html += f"            <tr>\n"
        html += f"                <td><strong>{escape(model)}</strong></td>\n"
        html += f"                <td class='avg-cell'>{overall_rate:.0f}%</td>\n"
        
        for result in results:
            if result is None:
                html += "                <td>N/A</td>\n"
            elif result['success']:
                html += "                <td class='success'>✅ Success</td>\n"
            else:
                html += "                <td class='failed'>❌ Failed</td>\n"
        
        html += "            </tr>\n"
    
    html += """        </tbody>
    </table>
</body>
</html>
"""
    
    return html


def format_comparison_console(model_data: Dict[str, List[dict]], run_labels: List[str]) -> str:
    """Generate a console-formatted comparison report."""
    lines = [
        "=" * 100,
        "  LLM BENCHMARK HISTORICAL COMPARISON",
        "=" * 100,
        "",
        "  RESPONSE TIME COMPARISON",
        "  " + "-" * 98,
    ]
    
    # Calculate column widths
    model_width = max(len(m) for m in model_data.keys()) + 2
    model_width = max(model_width, 20)
    
    # Header
    header = f"  {'Model':<{model_width}} {'Avg':<8}"
    for label in run_labels:
        header += f" {label:<12}"
    if len(run_labels) > 1:
        for i in range(len(run_labels) - 1):
            header += f" Trend{i+1:<8}"
    lines.append(header)
    lines.append("  " + "-" * 98)
    
    # Rows
    for model in sorted(model_data.keys()):
        results = model_data[model]
        
        successful_times = [r['response_time'] for r in results if r and r['success']]
        avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        row = f"  {model:<{model_width}} {avg_time:<8.2f}"
        
        for result in results:
            if result is None:
                row += f" {'N/A':<12}"
            elif result['success']:
                row += f" {result['response_time']:<11.2f}s"
            else:
                row += f" {'❌':<12}"
        
        for i in range(len(results) - 1):
            curr = results[i]
            next_res = results[i + 1]
            
            if curr is None or next_res is None:
                row += f" {'N/A':<10}"
            elif not curr['success'] or not next_res['success']:
                curr_success = curr['success'] if curr else False
                next_success = next_res['success'] if next_res else False
                arrow, status = calculate_success_trend(next_success, curr_success)
                status_short = status[:7]
                row += f" {arrow} {status_short:<7}"
            else:
                arrow, direction, pct = calculate_trend(next_res['response_time'], curr['response_time'])
                row += f" {arrow} {pct:<7.1f}%"
        
        lines.append(row)
    
    lines.extend([
        "  " + "-" * 98,
        "",
        "  Legend: ↓ = Faster (better), ↑ = Slower (worse), → = No change",
        "",
        "  SUCCESS RATE COMPARISON",
        "  " + "-" * 98,
    ])
    
    # Success rate header
    header = f"  {'Model':<{model_width}} {'Overall':<8}"
    for label in run_labels:
        header += f" {label:<12}"
    lines.append(header)
    lines.append("  " + "-" * 98)
    
    for model in sorted(model_data.keys()):
        results = model_data[model]
        successful = sum(1 for r in results if r and r['success'])
        overall_rate = (successful / len(results)) * 100 if results else 0
        
        row = f"  {model:<{model_width}} {overall_rate:<7.0f}%"
        
        for result in results:
            if result is None:
                row += f" {'N/A':<12}"
            elif result['success']:
                row += f" {'✅':<12}"
            else:
                row += f" {'❌':<12}"
        
        lines.append(row)
    
    lines.extend([
        "  " + "-" * 98,
        "",
        "=" * 100,
    ])
    
    return "\n".join(lines)


def generate_comparison_report(results_list: List[dict], format_type: str = "markdown") -> str:
    """Generate a historical comparison report from multiple benchmark runs."""
    if len(results_list) < 2:
        return "Error: Need at least 2 result files to compare."
    
    model_data, run_labels = build_comparison_data(results_list)
    
    if format_type == "markdown":
        return format_comparison_markdown(model_data, run_labels)
    elif format_type == "html":
        return format_comparison_html(model_data, run_labels)
    else:  # console
        return format_comparison_console(model_data, run_labels)


def save_report(content: str, output_dir: str, format_type: str) -> str:
    """Save the report to the specified output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = "md" if format_type == "markdown" else format_type
    filename = f"benchmark_report_{timestamp}.{extension}"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Generate formatted reports from LLM benchmark results"
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        help="Path to the JSON results file from benchmark.py"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "console"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save the report (default: results)"
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Include side-by-side comparison view"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Specific output file path (overrides default naming)"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="FILE",
        help="Compare multiple benchmark result files (e.g., --compare run1.json run2.json run3.json)"
    )
    
    args = parser.parse_args()
    
    # Handle comparison mode
    if args.compare:
        if len(args.compare) < 2:
            print("Error: --compare requires at least 2 result files.")
            sys.exit(1)
        
        results_list = load_multiple_results(args.compare)
        
        if len(results_list) < 2:
            print("Error: Could not load at least 2 valid result files for comparison.")
            sys.exit(1)
        
        report = generate_comparison_report(results_list, args.format)
        
        if args.format == "console":
            print(report)
        else:
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(report)
                print(f"Comparison report saved to: {output_path}")
            else:
                output_file = save_report(report, args.output_dir, args.format)
                print(f"Comparison report saved to: {output_file}")
        return
    
    # Standard single-file mode
    if not args.results_file:
        parser.print_help()
        sys.exit(1)
    
    # Load results (auto-detect format based on file extension)
    try:
        if args.results_file.lower().endswith('.csv'):
            data = load_results_csv(args.results_file)
        else:
            data = load_results(args.results_file)
    except FileNotFoundError:
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in results file: {e}")
        sys.exit(1)
    
    # Generate report based on format
    if args.format == "markdown":
        report = generate_markdown(data, args.side_by_side)
    elif args.format == "html":
        report = generate_html(data, args.side_by_side)
    else:  # console
        report = generate_console(data, args.side_by_side)
    
    # Output the report
    if args.format == "console":
        print(report)
    else:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        else:
            output_file = save_report(report, args.output_dir, args.format)
            print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
