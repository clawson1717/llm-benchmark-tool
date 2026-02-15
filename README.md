# LLM Benchmark Tool

A Python tool for benchmarking multiple Large Language Models (LLMs) simultaneously via the OpenRouter API. This tool allows you to send the same prompt to multiple models in parallel and compare their responses.

## Features

- **Parallel Execution**: Send prompts to multiple models concurrently for faster benchmarking
- **Multiple Model Support**: Compare responses from different LLM providers and models
- **Structured Output**: Results are saved as JSON with metadata including response times
- **CSV Export**: Export benchmark results as CSV for easy analysis in Excel/Google Sheets
- **Configurable**: Easy-to-use configuration files for defining which models to test
- **OpenRouter Integration**: Access to 200+ models through a single API
- **Automatic Retries**: Exponential backoff retry logic for transient network errors

## Use Cases

- **Model Comparison**: Compare different models for the same task
- **Cost Analysis**: Evaluate cost vs. performance across models
- **Quality Assessment**: Subjective evaluation of model outputs
- **Benchmarking**: Systematic testing of prompt effectiveness

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenRouter API key**:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

3. **Run a benchmark**:
   ```bash
   python src/benchmark.py "Explain quantum computing in simple terms"
   ```

4. **View the results**:
   ```bash
   python src/view_results.py results/benchmark_20240115_143022.json
   ```

## Retry Logic

The benchmark tool includes automatic retry logic with exponential backoff to handle transient network issues gracefully.

### How It Works

When an API request fails with a transient error (timeout, connection error, or 5xx server error), the tool automatically retries the request with increasing delays between attempts:

- **Retry 1**: 1 second delay
- **Retry 2**: 2 seconds delay
- **Retry 3**: 4 seconds delay

### Configuring Retries

By default, the tool retries up to 3 times. You can customize this with the `--max-retries` option:

```bash
# Disable retries (0 retries)
python src/benchmark.py "Explain quantum computing" --max-retries 0

# Increase retries to 5 for unstable connections
python src/benchmark.py "Explain quantum computing" --max-retries 5
```

### Retry Behavior

**Retried errors** (transient):
- Timeouts
- Connection errors
- 5xx server errors

**Non-retried errors** (client errors):
- 4xx errors (invalid API key, bad request, etc.)
- Parse errors

Retry attempts are logged to console so you can see when retries occur.

## Provider Filtering

Filter models by provider to benchmark only specific LLM providers.

### List Available Providers

```bash
# Show all available providers and model counts
python src/benchmark.py --list-providers
```

### Filter by Provider

```bash
# Benchmark only OpenAI models
python src/benchmark.py "Explain quantum computing" --provider openai

# Benchmark multiple specific providers
python src/benchmark.py "Explain quantum computing" --provider openai --provider anthropic

# Combine with other options
python src/benchmark.py "Explain quantum computing" --provider google --format csv
```

Provider names are case-insensitive (e.g., `openai`, `OpenAI`, and `OPENAI` all work).

## CSV Export

The benchmark tool supports exporting results directly to CSV format for easy analysis in spreadsheet applications.

### Export as CSV

```bash
# Run benchmark and save results as CSV
python src/benchmark.py "Explain quantum computing in simple terms" --format csv
```

### CSV Output Format

The CSV file includes the following columns:
- **model** - The model ID (e.g., `openai/gpt-4o`)
- **status** - `success` or `failed`
- **response_time** - Response time in seconds
- **tokens_used** - Total tokens consumed (0 if failed)
- **response_preview** - First 200 characters of the response (or error message if failed)

### Viewing CSV Results

The `view_results.py` script automatically detects CSV files and can generate reports from them:

```bash
# Generate markdown report from CSV
python src/view_results.py results/benchmark_20240115_143022.csv

# Generate HTML report from CSV
python src/view_results.py results/benchmark_20240115_143022.csv --format html
```

**Note**: When loading from CSV, the original prompt is not available and will display as "Loaded from CSV (prompt not available)".

## Project Structure

```
llm-benchmark-tool/
├── src/                  # Source code
│   ├── benchmark.py      # Main benchmarking script
│   └── view_results.py   # Results viewer and report generator
├── config/               # Configuration files
│   └── models.json       # Model definitions
├── examples/             # Sample prompts
│   └── prompts.txt       # Example prompts
├── results/              # Output directory (gitignored)
└── requirements.txt      # Python dependencies
```

## Results Viewer

The `view_results.py` script generates formatted reports from benchmark JSON output.

### Basic Usage

```bash
# Generate markdown report (default)
python src/view_results.py results/benchmark_20240115_143022.json

# Generate HTML report
python src/view_results.py results/benchmark_20240115_143022.json --format html

# Pretty print to console
python src/view_results.py results/benchmark_20240115_143022.json --format console

# Include side-by-side comparison view
python src/view_results.py results/benchmark_20240115_143022.json --side-by-side
```

### Options

- `--format {markdown,html,console}` - Output format (default: markdown)
- `--output-dir DIR` - Directory to save reports (default: results)
- `--output FILE` - Specific output file path
- `--side-by-side` - Include full response comparison view

### Report Contents

All reports include:
- **Prompt Used** - The original benchmark prompt
- **Summary Statistics** - Success rate, average response time, fastest/slowest models, total tokens
- **Results Table** - Model responses with status, timing, and token counts
- **Failed Requests** - Details of any failed API calls (if applicable)

## Historical Comparison

Track model performance over time by comparing multiple benchmark runs.

### Comparing Multiple Runs

Use the `--compare` option to analyze trends across different benchmark runs:

```bash
# Compare 3 different runs
python src/view_results.py --compare results/run1.json results/run2.json results/run3.json

# Generate HTML comparison report
python src/view_results.py --compare results/*.json --format html --output comparison.html

# View comparison in console
python src/view_results.py --compare results/run1.json results/run2.json --format console
```

### Comparison Features

The historical comparison report shows:

- **Response Time Trends**: Track how model response times change between runs
  - **↓** = Faster response time (improvement)
  - **↑** = Slower response time (regression)
  - **→** = No change
  - Percentage changes shown for each transition

- **Success Rate Tracking**: Monitor model reliability over time
  - Overall success rate across all runs
  - Per-run success/failure indicators
  - Visual indicators for improved/degraded reliability

- **Average Performance**: Average response time across all successful runs for each model

### Example Output

```
Model                     Avg      Run 1        Run 2        Run 3        Trend1      Trend2
------------------------------------------------------------------------------------------
gpt-4o                    1.09s    1.23s        1.05s        0.98s        ↓ 14.6%     ↓ 6.7%
claude-3-opus             2.30s    2.45s        2.30s        2.15s        ↓ 6.1%      ↓ 6.5%
gemini-pro                1.77s    1.89s        ❌            1.65s        ↓ degraded  ↑ improved
```

## Configuration

Edit `config/models.json` to customize which models to benchmark. The default configuration includes popular models like GPT-4, Claude, and Llama.

## Requirements

- Python 3.8+
- OpenRouter API key (get one at https://openrouter.ai/)

## License

MIT
