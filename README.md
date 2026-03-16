# factreasoner-mcp

An [MCP](https://modelcontextprotocol.io) server that exposes [IBM FactReasoner](https://github.com/IBM/FactReasoner) as a tool for LLM clients such as [Claude Code](https://github.com/anthropics/claude-code). FactReasoner is a probabilistic factuality assessment framework that decomposes LLM responses into atomic claims and verifies them against external knowledge sources using a Markov network inference engine ([Merlin](https://github.com/radum2275/merlin)).

This server uses a [LiteLLM](https://github.com/BerriAI/litellm) backend, making it usable with any LiteLLM-compatible provider. The default model is Llama 3.3 70B Instruct via Watsonx.

---

## Prerequisites

- Python >= 3.10
- [`uv`](https://github.com/astral-sh/uv) for environment management
- [Merlin](https://github.com/radum2275/merlin) compiled locally (C++ probabilistic inference engine)
- Watsonx credentials (or another LiteLLM-compatible provider)
- A [Serper API key](https://serper.dev) if using Google Search retrieval

---

## Installation

### 1. Clone this repo

```bash
git clone https://github.com/jbrry/factreasoner-mcp
cd factreasoner-mcp
```

### 2. Create the environment

```bash
uv venv --python 3.12
uv sync
```

`uv sync` will automatically pull the required dependencies directly from GitHub:
- `FactReasoner` from `IBM/FactReasoner` (`fix/litellm-backend-chat-key` branch)
- `litellm` from `jbrry/litellm` (`feat/top_logprobs_array` branch)

### 3. Configure environment variables

```bash
cp .env.example .env
# fill in your credentials
```

Required variables:

| Variable | Description |
|----------|-------------|
| `WATSONX_API_KEY` | IBM Watsonx API key |
| `WATSONX_URL` | Watsonx endpoint (e.g. `https://us-south.ml.cloud.ibm.com`) |
| `WATSONX_PROJECT_ID` | Watsonx project ID |
| `MERLIN_PATH` | Absolute path to the compiled `merlin` binary |
| `SERPER_API_KEY` | Serper API key (only required for `retriever_type="google"`) |

Optional overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTREASONER_MODEL_ID` | `watsonx/meta-llama/llama-3-3-70b-instruct` | Any LiteLLM model ID |
| `MAX_CONCURRENT_LLM_REQUESTS` | `4` | Concurrency cap to avoid rate limits |

### 4. Register with Claude Code
The below adds the command and server into the Claude client.
```bash
claude mcp add factreasoner-mcp -- \
  /path/to/factreasoner-mcp/.venv/bin/python \
  /path/to/factreasoner-mcp/server.py
```

See also [this example](https://modelcontextprotocol.io/examples#configuring-with-claude) for doing this manually via the settings file.

---

## Tools

### `assess_factuality`

Runs the full FactReasoner pipeline on a query/response pair.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | — | The original prompt given to the LLM |
| `response` | str | — | The LLM-generated response to assess |
| `topic` | str | — | Main subject (guides retrieval) |
| `retriever_type` | str | `"wikipedia"` | `"wikipedia"` or `"google"` |
| `model_id` | str | `watsonx/meta-llama/llama-3-3-70b-instruct` | LiteLLM model identifier |
| `pipeline_version` | str | `"FR2"` | `"FR1"`, `"FR2"`, or `"FR3"` |
| `top_k` | int | `5` | Contexts retrieved per atom |

---

## Pipeline Versions

| Version | Description |
|---------|-------------|
| FR1 | Each atom connected only to its own retrieved contexts |
| FR2 | All atoms connected to all unique contexts (default) |
| FR3 | FR2 + context-to-context relationships |

---

## Example

```
Query: How long do brines with low water activity (~0.5) reside under Martian surface conditions?

Response: At lower water activities (~0.5), brines may persist for up to approximately
1000 hours over the course of a Martian year.
```

---

## Citation

If you use this server in your work, please cite the FactReasoner paper:

```bibtex
@misc{marinescu2025factreasonerprobabilisticapproachlongform,
    title={FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models},
    author={Radu Marinescu and Debarun Bhattacharjya and Junkyu Lee and Tigran Tchrakian
            and Javier Carnerero Cano and Yufang Hou and Elizabeth Daly and Alessandra Pascale},
    year={2025},
    eprint={2502.18573},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2502.18573},
}
```
