"""
RAG Module Orchestrator

Main component that orchestrates the Retrieval-Augmented Generation pipeline.
Combines retrieval, prompt engineering, LLM generation, and citation management.
"""

from typing import List, Dict, Optional
import logging
import time

from .llm_provider import LLMProvider
from .prompt_templates import PromptTemplateFactory
from .citations import CitationExtractor
from .output_parser import OutputParser, RAGResponse

logger = logging.getLogger(__name__)


class RAGModule:
    """
    Orchestrates Retrieval-Augmented Generation pipeline.
    
    Pipeline Flow:
    1. Query input
    2. Prompt generation from query + documents
    3. LLM generation
    4. Citation extraction and enrichment
    5. Structured response output
    """

    def __init__(
        self,
        llm: LLMProvider,
        template_type: str = "domain_specific",
        parser: Optional[OutputParser] = None,
        pipeline=None,
    ):
        """
        Initialize RAG module.

        Args:
            llm: LLMProvider implementation (e.g., OllamaProvider)
            template_type: Prompt template strategy
                Options: "basic", "domain_specific", "chain_of_thought"
            parser: Custom OutputParser (optional)
            pipeline: SRIPipeline instance for automatic document retrieval (optional)
                When provided, enables autonomous retrieval without explicit documents

        Example:
            >>> from rag.llm_provider import OllamaProvider
            >>> llm = OllamaProvider()
            >>> rag = RAGModule(llm, template_type="chain_of_thought", pipeline=my_pipeline)
        """
        self.llm = llm
        self.parser = parser or OutputParser()
        self.pipeline = pipeline

        # Create prompt template
        self.template_factory = PromptTemplateFactory()
        self.template = self.template_factory.create(template_type)
        self.template_type = template_type

        pipeline_info = " with SRIPipeline integration" if pipeline else " in manual mode"
        logger.info(
            f"Initialized RAGModule with {template_type} template and "
            f"{llm.get_metadata().get('provider', 'unknown')} LLM{pipeline_info}"
        )

    def generate(
        self,
        query: str,
        documents: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
    ) -> RAGResponse:
        """
        Generate answer with optional retrieved documents.

        Implements hybrid integration pattern:
        - If documents provided: Use for augmenting generation
        - If no documents: Retrieve via pipeline

        Args:
            query: User question
            documents: Retrieved documents (optional). If provided, used to augment
                generation. Each document should have keys: id, title, content, url
            temperature: LLM creativity (0.0=deterministic, 1.0=random)
            max_tokens: Maximum output length
            top_p: Nucleus sampling parameter

        Returns:
            RAGResponse with generated answer and extracted citations

        Example:
            >>> query = "How does LSI work?"
            >>> response = rag.generate(query, documents=retrieved_docs)
            >>> print(response.answer)
            >>> for citation in response.citations:
            ...     print(f"  - {citation.title}")
        """
        start_time = time.time()

        # Auto-retrieval: Use pipeline if documents not provided
        if documents is None:
            if self.pipeline:
                try:
                    documents = self.pipeline.search(query, top_k=10)
                    logger.info(f"Retrieved {len(documents) if documents else 0} documents via SRIPipeline")
                except Exception as e:
                    logger.warning(f"Pipeline retrieval failed: {e}. Using pure generation mode.")
                    documents = []
            else:
                documents = []
                logger.debug("No documents provided and no pipeline - using pure generation mode")

        # Transform pipeline results to RAG document format
        documents = [
            {
                "id": result.get("doc_id") or result.get("id", "unknown"),
                "title": result.get("title", "Untitled"),
                "content": result.get("content", ""),
                "url": result.get("url"),
                "score": result.get("score"),
                "source": result.get("source", "local"),
            }
            for result in (documents if documents else [])
        ]

        logger.info(
            f"Generating answer for query: '{query[:60]}...' "
            f"with {len(documents)} documents"
        )

        try:
            # Step 1: Apply prompt template
            prompt = self.template.apply(query, documents)[0]
            prompt_time = time.time() - start_time

            logger.debug(f"Prompt generated in {prompt_time:.2f}s ({len(prompt)} chars)")

            # Step 2: Generate LLM response
            llm_start = time.time()
            raw_response = self.llm.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            llm_time = time.time() - llm_start

            logger.debug(f"LLM generation completed in {llm_time:.2f}s")

            # Step 3: Parse output to structured format
            parse_start = time.time()
            # Pass documents to parser for citation enrichment
            rag_response = self._parse_response(raw_response, documents)
            
            parse_time = time.time() - parse_start

            # Step 4: Recover citations from answer text if parser returned none
            citations_start = time.time()
            if not rag_response.citations and rag_response.answer and documents:
                answer, citations = CitationExtractor.extract_citations(
                    rag_response.answer, documents
                )
                rag_response = RAGResponse(answer=answer, citations=citations)

            citations_time = time.time() - citations_start

            # Step 5: Validate response
            is_valid = self.parser.validate(rag_response)

            total_time = time.time() - start_time

            # Log performance metrics
            logger.info(
                f"RAG generation completed in {total_time:.2f}s\n"
                f"  - Prompt: {prompt_time:.2f}s\n"
                f"  - LLM: {llm_time:.2f}s ({(llm_time/total_time*100):.1f}%)\n"
                f"  - Parse: {parse_time:.2f}s\n"
                f"  - Citations: {citations_time:.2f}s\n"
                f"  - Valid: {is_valid}\n"
                f"  - Citations: {len(rag_response.citations)}"
            )

            return rag_response

        except Exception as e:
            logger.error(f"RAG generation failed: {e}", exc_info=True)
            # Return error response
            return RAGResponse(
                answer=f"Error generating response: {str(e)}",
                citations=[],
            )

    def _parse_response(self, raw_response: str, documents: Optional[List[Dict]] = None) -> RAGResponse:
        """
        Parse raw LLM response using OutputParser.

        Args:
            raw_response: Raw text from LLM
            documents: Retrieved documents for citation enrichment in fallback mode

        Returns:
            Parsed RAGResponse

        Raises:
            Exception: If parsing fails completely
        """
        return self.parser.parse(raw_response, documents)

    def switch_template(self, template_type: str) -> None:
        """
        Switch prompt template at runtime.

        Useful for A/B testing different prompt strategies.

        Args:
            template_type: Template name
                ("basic", "domain_specific", "chain_of_thought")

        Example:
            >>> rag.switch_template("chain_of_thought")
        """
        try:
            self.template = self.template_factory.create(template_type)
            self.template_type = template_type
            logger.info(f"Switched to template: {template_type}")
        except ValueError as e:
            logger.error(f"Template switch failed: {e}")
            raise

    def get_metadata(self) -> Dict:
        """Return RAG module metadata for debugging.

        Returns:
            Dictionary with configuration and statistics
        """
        return {
            "module": "RAG",
            "template": self.template_type,
            "llm": self.llm.get_metadata(),
            "available_templates": self.template_factory.available_templates(),
        }
