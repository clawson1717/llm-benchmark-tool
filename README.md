# LLM Benchmark Tool

A Python tool for benchmarking multiple Large Language Models (LLMs) simultaneously via the OpenRouter API. This tool allows you to send the same prompt to multiple models in parallel and compare their responses.

## Features

- **Parallel Execution**: Send prompts to multiple models concurrently for faster benchmarking
- **Multiple Model Support**: Compare responses from different LLM providers and models
- **Structured Output**: Results are saved as JSON with metadata including response times
- **CSV Export**: Export benchmark results as CSV for easy analysis in Excel/Google Sheets
- **Configurable**: Easy-to-use configuration files for defining which models to test
- **OpenRouter Integration**: Access to 200+ models through a single API

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

## Configuration

Edit `config/models.json` to customize which models to benchmark. The default configuration includes popular models like GPT-4, Claude, and Llama.

## Requirements

- Python 3.8+
- OpenRouter API key (get one at https://openrouter.ai/)

## License

MIT
