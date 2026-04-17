from src.types import ModelLimits

DEFAULT_THRESHOLD = 0.8

MODEL_LIMITS: dict[str, ModelLimits] = {
    "gpt-5": ModelLimits(
        input_limit=272_000,
        output_limit=128_000,
        context_window=400_000,
    ),
    "gpt-5-mini": ModelLimits(
        input_limit=272_000,
        output_limit=128_000,
        context_window=400_000,
    ),
}

DEFAULT_LIMITS = ModelLimits(
    input_limit=128_000,
    output_limit=16_000,
    context_window=128_000,
)


def get_model_limits(model: str) -> ModelLimits:
    """Get token limits for a specific model."""
    if model in MODEL_LIMITS:
        return MODEL_LIMITS[model]
    if model.startswith("gpt-5"):
        return MODEL_LIMITS["gpt-5"]
    return DEFAULT_LIMITS


def is_over_threshold(
    total_tokens: int,
    context_window: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Check if token usage exceeds the threshold."""
    return total_tokens > context_window * threshold


def calculate_usage_percentage(total_tokens: int, context_window: int) -> float:
    """Calculate usage percentage."""
    return (total_tokens / context_window) * 100
