"""FactReasoner MCP server using LiteLLM backend."""

import asyncio
import os
from pathlib import Path
from typing import Any

import nest_asyncio

# Allow FactReasoner's asyncio.run() calls to nest inside FastMCP's event loop.
# Without this, asyncio.run() raises "cannot be called from a running event loop"
# and workarounds that create separate loops cause "Event loop is closed" /
# "bound to a different event loop" errors in litellm's cached async clients.
nest_asyncio.apply()

from dotenv import load_dotenv
from fact_reasoner import FactReasoner
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import ContextRetriever, Retriever
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer
from mcp.server.fastmcp import FastMCP
from mellea.backends.litellm import LiteLLMBackend
from mellea.backends.model_options import ModelOption

import litellm

load_dotenv(Path(__file__).parent / ".env")

# Cap concurrent LiteLLM requests to avoid Watsonx rate limits (8 req/s).
# FactReasoner fires many NLI calls concurrently via asyncio.gather — the
# semaphore throttles parallelism without modifying FactReasoner internals.
_MAX_CONCURRENT_LLM_REQUESTS = int(os.environ.get("MAX_CONCURRENT_LLM_REQUESTS", "4"))
_llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM_REQUESTS)
_original_acompletion = litellm.acompletion


async def _throttled_acompletion(*args, **kwargs):
    async with _llm_semaphore:
        return await _original_acompletion(*args, **kwargs)


litellm.acompletion = _throttled_acompletion

mcp = FastMCP("fact_reasoner")

MERLIN_PATH = os.environ.get(
    "MERLIN_PATH",
    "./lib/merlin/build/merlin",
)

# Default model: Watsonx Granite 3.3 8B via LiteLLM.
# Override with FACTREASONER_MODEL_ID env var.
DEFAULT_MODEL_ID = os.environ.get(
    "FACTREASONER_MODEL_ID",
    "watsonx/meta-llama/llama-3-3-70b-instruct",
)


def _build_backend(model_id: str) -> LiteLLMBackend:
    """Create a LiteLLM backend. For Watsonx models, base_url is read from
    WATSONX_URL env var by LiteLLM automatically, so we pass None."""
    base_url = None if model_id.startswith("watsonx/") else os.environ.get("LITELLM_BASE_URL")
    return LiteLLMBackend(
        model_id=model_id,
        base_url=base_url,
        model_options={ModelOption.MAX_NEW_TOKENS: 4096},
    )


def _build_pipeline(
    backend: LiteLLMBackend,
    retriever_type: str = "google",
    top_k: int = 3,
    cache_dir: str | None = None,
) -> FactReasoner:
    """Assemble a FactReasoner pipeline from a backend."""
    qb = QueryBuilder(backend)
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    retriever = Retriever(
        service_type=retriever_type,
        top_k=top_k,
        cache_dir=cache_dir,
        fetch_text=(retriever_type == retriever_type),
        query_builder=qb,
        num_workers=4,
    )
    context_retriever = ContextRetriever(retriever=retriever, num_workers=4)
    context_summarizer = ContextSummarizer(backend)
    nli_extractor = NLIExtractor(backend)

    return FactReasoner(
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        context_retriever=context_retriever,
        context_summarizer=context_summarizer,
        nli_extractor=nli_extractor,
        merlin_path=MERLIN_PATH,
    )


def _run_assess_factuality(
    query: str,
    response: str,
    topic: str,
    retriever_type: str,
    model_id: str,
    pipeline_version: str,
    top_k: int,
) -> dict[str, Any]:
    rel_context_context = pipeline_version == "FR3"
    remove_duplicates = pipeline_version in ("FR2", "FR3")

    backend = _build_backend(model_id)
    pipeline = _build_pipeline(backend, retriever_type=retriever_type, top_k=top_k)

    pipeline.build(
        query=query,
        response=response,
        topic=topic,
        has_atoms=False,
        has_contexts=False,
        revise_atoms=True,
        remove_duplicates=remove_duplicates,
        summarize_contexts=False,
        rel_atom_context=True,
        rel_context_context=rel_context_context,
        use_fast_retriever=True,
    )

    results, marginals = pipeline.score()
    return {"results": results, "marginals": marginals}


@mcp.tool()
async def assess_factuality(
    query: str,
    response: str,
    topic: str,
    retriever_type: str = "wikipedia",
    model_id: str = DEFAULT_MODEL_ID,
    pipeline_version: str = "FR2",
    top_k: int = 5,
) -> dict[str, Any]:
    """Assess the factuality of an LLM response using probabilistic reasoning.

    Runs the full FactReasoner pipeline: atomic decomposition → decontextualization
    → context retrieval → NLI verification → Markov network inference (Merlin).

    Args:
        query: The original question or prompt given to the LLM.
        response: The LLM-generated response to assess.
        topic: The main subject of the response (used to guide retrieval).
        retriever_type: Knowledge source — "wikipedia" (default), "google" (needs SERPER_API_KEY).
        model_id: LiteLLM model identifier, e.g. "watsonx/meta-llama/llama-3-3-70b-instruct".
        pipeline_version: "FR1" (atom↔own contexts), "FR2" (atom↔all contexts, default),
                          or "FR3" (FR2 + context↔context).
        top_k: Number of contexts to retrieve per atom.
    """
    return _run_assess_factuality(
        query, response, topic, retriever_type, model_id, pipeline_version, top_k,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
