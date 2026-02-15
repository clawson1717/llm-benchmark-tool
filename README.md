# LLM Benchmark Tool

A Python tool for benchmarking multiple Large Language Models (LLMs) simultaneously via the OpenRouter API. This tool allows you to send the same prompt to multiple models in parallel and compare their responses.

## Features

- **Parallel Execution**: Send prompts to multiple models concurrently for faster benchmarking
- **Multiple Model Support**: Compare responses from different LLM providers and models
- **Structured Output**: Results are saved as JSON with metadata including response times
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

## Project Structure

```
llm-benchmark-tool/
├── src/              # Source code
│   └── benchmark.py  # Main benchmarking script
├── config/           # Configuration files
│   └── models.json   # Model definitions
├── examples/         # Sample prompts
│   └── prompts.txt   # Example prompts
├── results/          # Output directory (gitignored)
└── requirements.txt  # Python dependencies
```

## Configuration

Edit `config/models.json` to customize which models to benchmark. The default configuration includes popular models like GPT-4, Claude, and Llama.

## Requirements

- Python 3.8+
- OpenRouter API key (get one at https://openrouter.ai/)

## License

MIT
