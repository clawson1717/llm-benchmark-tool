# LLM Benchmark Historical Comparison

## Response Time Comparison

| Model | Avg | 2025-02-01 | 2025-02-10 | 2025-02-15 | Trend1 | Trend2 |
|-------|-----|---------|---------|---------|---------|---------|
| anthropic/claude-3-opus | 2.30s | 2.45s | 2.30s | 2.15s | ↓ 6.1% | ↓ 6.5% |
| google/gemini-pro | 1.77s | 1.89s | ❌ | 1.65s | ↓ degraded | ↑ improved |
| openai/gpt-4o | 1.09s | 1.23s | 1.05s | 0.98s | ↓ 14.6% | ↓ 6.7% |

### Legend
- **↓** = Faster response time (improvement)
- **↑** = Slower response time (regression)
- **→** = No change
- **Avg** = Average response time across all successful runs

## Success Rate Comparison

| Model | Overall | 2025-02-01 | 2025-02-10 | 2025-02-15 |
|-------|---------|-------|-------|-------|
| anthropic/claude-3-opus | 100% | ✅ | ✅ | ✅ |
| google/gemini-pro | 67% | ✅ | ❌ | ✅ |
| openai/gpt-4o | 100% | ✅ | ✅ | ✅ |