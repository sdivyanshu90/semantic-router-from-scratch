"""Custom exceptions raised by the semantic router package."""


class SemanticRouterError(Exception):
    """Base exception for all semantic router failures."""


class RouteNotEmbeddedError(SemanticRouterError):
    """Raised when a route is scored before its embeddings are prepared."""


class RouteNotFoundError(SemanticRouterError):
    """Raised when a requested route name does not exist."""


class RouteConfigurationError(SemanticRouterError):
    """Raised when a route or router is configured with invalid values."""


class EncoderError(SemanticRouterError):
    """Raised when an embedding backend cannot fulfill a request."""


class CalibrationError(SemanticRouterError):
    """Raised when threshold calibration cannot complete successfully."""
