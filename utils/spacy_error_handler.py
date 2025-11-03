"""
SpaCy Error Handler for Robust NLP Processing

This module provides centralized error handling for spaCy operations in the nith_rag system.
It implements graceful degradation strategies and comprehensive logging for all spaCy-related
errors, ensuring the system remains functional even when spaCy is unavailable or encounters issues.

Key Features:
- SpaCy model loading with comprehensive error handling
- Document encoding validation and error logging
- Graceful fallback to regex-based tokenization
- Structured logging for monitoring and debugging
- Performance-optimized error recovery strategies

Usage:
    from utils.spacy_error_handler import SpaCyErrorHandler

    handler = SpaCyErrorHandler(model_name="en_core_web_sm")
    nlp = handler.load_model()

    if nlp is None:
        # Use regex fallback
        handler.logger.warning("SpaCy unavailable, using regex fallback")

    # Process text with error handling
    processed = handler.process_text(text, operation="tokenize")
"""

from loguru import logger
import time
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager
from enum import Enum
import json

# Try spaCy imports with fallback
try:
    import spacy
    from spacy.lang.en import English
    from spacy.tokens import Doc

    SPACY_AVAILABLE = True
except ImportError as e:
    spacy = None
    English = None
    Doc = None
    SPACY_AVAILABLE = False
    SPACY_IMPORT_ERROR = str(e)

# Using Loguru's global logger; instance-specific context is bound per-instance


class SpaCyError(Enum):
    """Enumeration of spaCy-related error types for systematic handling."""

    IMPORT_ERROR = "import_error"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    PIPELINE_ERROR = "pipeline_error"
    ENCODING_ERROR = "encoding_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    UNSUPPORTED_MODEL = "unsupported_model"


class SpaCyErrorHandler:
    """
    Centralized spaCy error handling and fallback management.

    This class provides robust spaCy operations with comprehensive error handling,
    fallback strategies, and structured logging for monitoring system health.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        fallback_model: Optional[str] = None,
        load_timeout: float = 30.0,
        enable_logging: bool = True,
        log_level: str = "DEBUG",
    ):
        """
        Initialize SpaCyErrorHandler with configuration.

        Args:
            model_name: Primary spaCy model to load (default: en_core_web_sm)
            fallback_model: Alternative model to try if primary fails
            load_timeout: Maximum time to wait for model loading (seconds)
            enable_logging: Whether to enable structured logging
            log_level: Logging level for spaCy operations
        """
        self.model_name = model_name
        self.fallback_model = fallback_model or "en_core_web_md"
        self.load_timeout = load_timeout
        self.enable_logging = enable_logging
        # Store desired log level name for potential centralized configuration
        self.log_level = log_level.upper()

        # State tracking
        self.nlp: Optional[Any] = None
        self.fallback_active = False
        self.errors_encountered: List[Dict[str, Any]] = []
        self.performance_stats = {
            "model_load_time": None,
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
        }

        # Performance thresholds for monitoring
        self.memory_threshold_mb = 500  # Warning threshold for memory usage

        # Setup custom logger
        self.logger = self._setup_logger()

        # Log initialization
        self._log_initialization()

    def _setup_logger(self):
        """Return a Loguru-bound logger for this SpaCy handler instance.

        We avoid adding sinks here to prevent duplicate outputs when multiple
        instances are created. Global sinks and formatting should be configured
        centrally (e.g., in `main.py` or `utils/logging_setup.py`).
        """
        bound_logger = logger.bind(
            component="spacy_error_handler",
            instance_id=id(self),
            model_name=self.model_name,
        )

        return bound_logger

    def _log_initialization(self) -> None:
        """Log initialization parameters for debugging."""
        init_info = {
            "spacy_available": SPACY_AVAILABLE,
            "model_name": self.model_name,
            "fallback_model": self.fallback_model,
            "load_timeout": self.load_timeout,
        }

        if not SPACY_AVAILABLE:
            init_info["import_error"] = SPACY_IMPORT_ERROR
            # Loguru accepts arbitrary kwargs which appear under record.extra
            self.logger.warning(
                "spaCy not available, will use regex fallback",
                spacy_error=SPACY_IMPORT_ERROR,
            )

        self.logger.info("SpaCyErrorHandler initialized", **init_info)

    @contextmanager
    def operation_context(self, operation: str, text_length: int = 0):
        """Context manager for spaCy operations with performance tracking."""
        start_time = time.perf_counter()
        operation_id = f"{operation}_{id(self)}_{int(start_time * 1000)}"

        self._log_operation_start(operation, operation_id, text_length)

        try:
            yield operation_id
            self.performance_stats["successful_operations"] += 1
            self._log_operation_success(operation, operation_id, start_time)

        except Exception as e:
            self.performance_stats["failed_operations"] += 1
            self._handle_operation_error(e, operation, operation_id)
            raise e

        finally:
            self.performance_stats["total_operations"] += 1

    def load_model(self) -> Optional[Any]:
        """
        Load spaCy model with comprehensive error handling.

        Returns:
            Loaded spaCy model or None if loading fails

        Raises:
            RuntimeError: If model loading completely fails and no fallback available
        """
        if not SPACY_AVAILABLE:
            self._handle_import_error()
            return None

        load_start = time.perf_counter()

        # Try primary model
        try:
            with self.operation_context("model_load", 0):
                self.logger.info(f"Loading spaCy model: {self.model_name}")
                self.nlp = spacy.load(self.model_name)

                # Ensure sentence boundary component (sentencizer) is present.
                try:
                    if "sentencizer" not in getattr(self.nlp, "pipe_names", []):
                        # Prefer adding by string name; fall back to Sentencizer class if needed
                        try:
                            self.nlp.add_pipe("sentencizer")
                            self.logger.info("Added 'sentencizer' to spaCy pipeline")
                        except Exception:
                            try:
                                from spacy.pipeline import Sentencizer

                                self.nlp.add_pipe(Sentencizer())
                                self.logger.info(
                                    "Added Sentencizer() to spaCy pipeline"
                                )
                            except Exception as se:
                                self.logger.warning(
                                    "Could not add sentencizer to spaCy pipeline",
                                    error=str(se),
                                )
                except Exception:
                    # Non-fatal: continue even if we cannot manipulate the pipeline
                    pass

                load_time = time.perf_counter() - load_start
                self.performance_stats["model_load_time"] = load_time

                self._log_model_success(self.model_name, load_time)
                return self.nlp

        except Exception as e:
            self._handle_model_load_error(e, self.model_name)

            # Try fallback model if available
            if self.fallback_model and self.fallback_model != self.model_name:
                try:
                    self.logger.warning(
                        f"Trying fallback model: {self.fallback_model}",
                        original_error=str(e),
                    )

                    with self.operation_context("fallback_model_load", 0):
                        self.nlp = spacy.load(self.fallback_model)

                        load_time = time.perf_counter() - load_start
                        self.performance_stats["model_load_time"] = load_time

                        self._log_model_success(
                            self.fallback_model, load_time, fallback=True
                        )
                        self.fallback_active = True
                        return self.nlp

                except Exception as fallback_error:
                    self._handle_fallback_failure(fallback_error)

            # All loading attempts failed
            self.logger.error(
                "All spaCy model loading attempts failed, system will use regex fallback",
                errors=[str(e) for e in self.errors_encountered],
                load_time=time.perf_counter() - load_start,
            )

            return None

    def process_text(
        self, text: str, operation: str = "process", validate_encoding: bool = True
    ) -> Optional[Any]:
        """
        Process text with spaCy with error handling.

        Args:
            text: Text to process
            operation: Type of operation (tokenize, sentencize, etc.)
            validate_encoding: Whether to validate text encoding first

        Returns:
            Processed result or None if processing fails
        """
        if self.nlp is None:
            self.logger.debug("No spaCy model available, skipping processing")
            return None

        if validate_encoding:
            encoding_valid, encoding_error = self._validate_text_encoding(text)
            if not encoding_valid:
                self._handle_encoding_error(encoding_error, operation, len(text))
                return None

        try:
            with self.operation_context(operation, len(text)):
                if operation == "sentencize":
                    # Simple sentence segmentation pipeline
                    doc = self.nlp(
                        text, disable=["tagger", "parser", "lemmatizer", "ner"]
                    )
                    return list(doc.sents)
                elif operation == "tokenize":
                    doc = self.nlp(text, disable=["parser", "lemmatizer", "ner"])
                    return [token.text for token in doc]
                else:
                    # Full processing
                    return self.nlp(text)

        except Exception as e:
            self._handle_processing_error(e, operation, len(text))
            return None

    def get_regex_tokenizer(self) -> Dict[str, Any]:
        """
        Get regex-based fallback tokenizer when spaCy is unavailable.

        Returns:
            Dictionary with tokenization functions and metadata
        """
        import re

        # Simple English tokenization pattern
        token_pattern = re.compile(r"\b\w+\b|'t|'re|'s|'d|'ll|'ve")

        def tokenize(text: str) -> List[str]:
            """Simple regex-based tokenization."""
            return token_pattern.findall(text)

        def count_tokens(text: str) -> int:
            """Count tokens in text."""
            return len(tokenize(text))

        def split_sentences(text: str) -> List[str]:
            """Simple sentence splitting using regex."""
            # Basic sentence patterns
            sentence_pattern = re.compile(r"(?<=[.!?])\s+")
            sentences = sentence_pattern.split(text.strip())

            # Filter out very short fragments
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            return sentences if sentences else [text.strip()]

        return {
            "tokenize": tokenize,
            "count_tokens": count_tokens,
            "split_sentences": split_sentences,
            "is_fallback": True,
            "description": "Regex-based tokenization fallback",
        }

    def _validate_text_encoding(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate text encoding and identify potential issues.

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for null bytes (common corruption indicator)
            if "\x00" in text:
                null_count = text.count("\x00")
                if null_count > len(text) * 0.001:  # More than 0.1% null bytes
                    return (
                        False,
                        f"Excessive null bytes ({null_count}) indicating corruption",
                    )

            # Try UTF-8 encoding/decoding roundtrip
            encoded = text.encode("utf-8", errors="replace")
            decoded = encoded.decode("utf-8")

            if decoded != text:
                return False, "UTF-8 encoding/decoding roundtrip failed"

            return True, None

        except Exception as e:
            return False, f"Encoding validation failed: {str(e)}"

    def _handle_import_error(self) -> None:
        """Handle spaCy import errors."""
        error_info = {
            "error_type": SpaCyError.IMPORT_ERROR.value,
            "message": "spaCy import failed",
            "details": SPACY_IMPORT_ERROR,
            "timestamp": time.time(),
        }
        self.errors_encountered.append(error_info)
        self.logger.error("spaCy import failed, using regex fallback", **error_info)

    def _handle_model_load_error(self, error: Exception, model_name: str) -> None:
        """Handle spaCy model loading errors."""
        error_type = self._classify_error(error)

        error_info = {
            "error_type": error_type.value,
            "message": f"Failed to load spaCy model: {model_name}",
            "model_name": model_name,
            "original_error": str(error),
            "error_class": error.__class__.__name__,
            "timestamp": time.time(),
        }
        self.errors_encountered.append(error_info)
        self.logger.error(f"spaCy model load failed: {model_name}", **error_info)

    def _handle_fallback_failure(self, error: Exception) -> None:
        """Handle fallback model loading failure."""
        error_info = {
            "error_type": SpaCyError.MODEL_LOAD_FAILED.value,
            "message": "Fallback model loading also failed",
            "original_error": str(error),
            "timestamp": time.time(),
        }
        self.errors_encountered.append(error_info)
        self.logger.error("Fallback model loading failed", **error_info)

    def _handle_encoding_error(
        self, error_msg: str, operation: str, text_length: int
    ) -> None:
        """Handle text encoding errors."""
        error_info = {
            "error_type": SpaCyError.ENCODING_ERROR.value,
            "message": error_msg,
            "operation": operation,
            "text_length": text_length,
            "timestamp": time.time(),
        }
        self.errors_encountered.append(error_info)
        self.logger.warning(f"Text encoding error in {operation}", **error_info)

    def _handle_processing_error(
        self, error: Exception, operation: str, text_length: int
    ) -> None:
        """Handle spaCy processing errors."""
        error_type = self._classify_error(error)

        error_info = {
            "error_type": error_type.value,
            "message": f"spaCy processing failed in {operation}",
            "operation": operation,
            "text_length": text_length,
            "original_error": str(error),
            "error_class": error.__class__.__name__,
            "timestamp": time.time(),
        }
        self.errors_encountered.append(error_info)
        self.logger.warning(f"spaCy processing error in {operation}", **error_info)

    def _handle_operation_error(
        self, error: Exception, operation: str, operation_id: str
    ) -> None:
        """Handle general operation errors in context manager."""
        error_type = self._classify_error(error)

        error_info = {
            "error_type": error_type.value,
            "message": f"Operation {operation} failed: {str(error)}",
            "operation": operation,
            "operation_id": operation_id,
            "original_error": str(error),
            "error_class": error.__class__.__name__,
            "timestamp": time.time(),
        }
        self.errors_encountered.append(error_info)
        self.logger.error(f"Operation error in {operation}", **error_info)

    def _classify_error(self, error: Exception) -> SpaCyError:
        """Classify exception type for better error handling."""
        error_str = str(error).lower()
        error_type = error.__class__.__name__

        if "model not found" in error_str or "can't find model" in error_str:
            return SpaCyError.MODEL_NOT_FOUND
        elif "timeout" in error_str or "timed out" in error_str:
            return SpaCyError.TIMEOUT_ERROR
        elif "memory" in error_str or "out of memory" in error_str:
            return SpaCyError.MEMORY_ERROR
        elif isinstance(error, UnicodeDecodeError) or "encoding" in error_str:
            return SpaCyError.ENCODING_ERROR
        else:
            return SpaCyError.PIPELINE_ERROR

    def _log_operation_start(
        self, operation: str, operation_id: str, text_length: int
    ) -> None:
        """Log operation start."""
        if self.enable_logging:
            self.logger.debug(
                f"Starting {operation}",
                operation=operation,
                operation_id=operation_id,
                text_length=text_length,
                model_loaded=self.nlp is not None,
                fallback_active=self.fallback_active,
            )

    def _log_operation_success(
        self, operation: str, operation_id: str, start_time: float
    ) -> None:
        """Log successful operation."""
        duration = time.perf_counter() - start_time

        if self.enable_logging:
            self.logger.debug(
                f"Completed {operation}",
                operation=operation,
                operation_id=operation_id,
                duration_seconds=round(duration, 4),
                performance_ok=duration < 1.0,  # Sub-second target
            )

    def _log_model_success(
        self, model_name: str, load_time: float, fallback: bool = False
    ) -> None:
        """Log successful model loading."""
        spacy_version = (
            getattr(spacy, "__version__", "unknown") if spacy else "unavailable"
        )
        self.logger.info(
            f"spaCy model loaded successfully: {model_name}",
            model_name=model_name,
            load_time_seconds=round(load_time, 4),
            is_fallback=fallback,
            fallback_active=self.fallback_active,
            spacy_version=spacy_version,
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        error_counts = {}
        for error in self.errors_encountered:
            error_type = error.get("error_type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {
            "total_errors": len(self.errors_encountered),
            "error_counts": error_counts,
            "recent_errors": (
                self.errors_encountered[-5:] if self.errors_encountered else []
            ),
            "spacy_available": SPACY_AVAILABLE,
            "model_loaded": self.nlp is not None,
            "fallback_active": self.fallback_active,
            "performance_stats": self.performance_stats,
        }
