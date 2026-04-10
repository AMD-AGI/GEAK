"""
RAG postprocessor for filtering and restructuring retrieval results.

This module uses an LLM to post-process RAG retrieval results:
filtering irrelevant chunks, deduplication, and reorganizing content
for downstream consumption.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from minisweagent.models.amd_llm import AmdLlmModel

logger = logging.getLogger(__name__)


@dataclass
class RAGPostProcessorConfig:
    """Configuration for RAG postprocessor behavior."""

    model_name: str = "claude-opus-4.6"
    api_key: str | None = None
    system_prompt: str | None = None
    enabled: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class RAGPostProcessor:
    """
    Post-processor for RAG retrieval results.

    Processes retrieved chunks from RAG queries by:
    1. Evaluating relevance
    2. Removing duplicates
    3. Reorganizing content for downstream LLM consumption
    """

    DEFAULT_SYSTEM_PROMPT = """You are a sub-agent in a code generation and optimization system.

Your role is to reorganize, clean, and structure retrieved RAG chunks so that
a downstream LLM can easily understand and reason over them.
You should preserve relevant technical details and only remove information
that is clearly out of scope.

Input:
- User query: [QUERY]
- Retrieved chunks: [CHUNKS]

Instructions:

1. Relevance filtering (conservative):
   - Keep all chunks that are clearly relevant or potentially useful.
   - Discard chunks only if they are clearly unrelated or outside the target domain.
   - Example: For HIP / ROCm tasks, discard Triton-only or unrelated framework content
     unless explicit relevance to HIP is stated.

2. Deduplication:
   - Remove exact duplicates.
   - Merge highly overlapping chunks while preserving all unique technical details.

3. Reorganization (not summarization):
   - Preserve code snippets, parameters, constraints, and technical nuances.
   - Reorganize content using clear sections, headings, and bullet points.
   - Group information by topic (e.g., APIs, examples, performance considerations, pitfalls).

4. Faithfulness and bounded augmentation:
   - You may add minimal explanatory or connective text to improve clarity or logical flow.
   - Added text must be directly implied by the chunks and must not introduce new
     technical facts, APIs, or claims.
   - Do not add speculative guidance or external knowledge.
   - Mark inferred or connective additions with "(clarification)" or "(inferred)".

5. Output rules:
   - Output only the cleaned, structured content.
   - Do not include reasoning steps, scores, or meta commentary.
   - If no relevant content remains, output exactly:
     "No relevant information found."
   """

    def __init__(self, config: RAGPostProcessorConfig | None = None):
        self.config = config or RAGPostProcessorConfig()
        self._model = None

    @property
    def model(self) -> AmdLlmModel:
        """Lazy initialization of the LLM model."""
        if self._model is None:
            self._model = AmdLlmModel(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                model_kwargs=self.config.model_kwargs,
            )
            self._model._impl.tools = []
        return self._model

    def process(self, rag_result: str, query: str = "") -> str:
        """
        Process RAG retrieval results through the postprocessor.

        Args:
            rag_result: Raw result from RAG retrieval
            query: Optional original query for context

        Returns:
            Filtered and reorganized result
        """
        if not self.config.enabled:
            return rag_result

        system_prompt = self.config.system_prompt or self.DEFAULT_SYSTEM_PROMPT

        user_content = rag_result
        if query:
            user_content = f"Query: {query}\n\n{rag_result}"

        logger.debug("RAG postprocessor processing %d chars", len(rag_result))

        response = self.model.query(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        )

        result = response["content"]
        logger.debug("RAG postprocessor output %d chars", len(result))

        return result


def create_rag_postprocessor(**kwargs) -> RAGPostProcessor:
    """Convenience function to create a RAG postprocessor."""
    config = RAGPostProcessorConfig(**kwargs)
    return RAGPostProcessor(config)
