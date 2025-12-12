"""Exceptions for the query processing pipeline."""


class QueryServiceError(Exception):
    """Base exception for query service errors."""

    def __init__(self, message: str, stage: str, http_status: int = 500):
        self.message = message
        self.stage = stage
        self.http_status = http_status
        super().__init__(message)


class QueryOptimizationError(QueryServiceError):
    """Error during query optimization."""

    def __init__(self, message: str):
        super().__init__(message, "query_optimization", 500)


class SemanticScholarError(QueryServiceError):
    """Error during Semantic Scholar API calls."""

    def __init__(self, message: str):
        super().__init__(message, "semantic_scholar_search", 503)


class QualityGateError(QueryServiceError):
    """Error during quality gate validation."""

    def __init__(self, message: str):
        super().__init__(message, "quality_gate", 500)


class RankingError(QueryServiceError):
    """Error during paper ranking."""

    def __init__(self, message: str):
        super().__init__(message, "composite_scoring", 500)


class AcquisitionError(QueryServiceError):
    """Error during PDF acquisition."""

    def __init__(self, message: str):
        super().__init__(message, "pdf_acquisition", 500)


class ProcessingError(QueryServiceError):
    """Error during processing and embedding."""

    def __init__(self, message: str):
        super().__init__(message, "processing", 500)


class RetrievalError(QueryServiceError):
    """Error during hybrid retrieval."""

    def __init__(self, message: str):
        super().__init__(message, "retrieval", 500)


class RerankingError(QueryServiceError):
    """Error during cross-encoder reranking."""

    def __init__(self, message: str):
        super().__init__(message, "reranking", 500)


class SynthesisError(QueryServiceError):
    """Error during answer synthesis."""

    def __init__(self, message: str):
        super().__init__(message, "synthesis", 500)


class VerificationError(QueryServiceError):
    """Error during citation verification."""

    def __init__(self, message: str):
        super().__init__(message, "verification", 500)
