import time
import logging
import json
from datetime import datetime
from functools import wraps
from models.state import State

logger = logging.getLogger(__name__)


def custom_json_serializer(obj):
    """Custom JSON serializer that handles datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def log_state_transition(node_name: str):
    """
    A decorator that logs the state before and after a node's execution,
    along with execution time and other metadata.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(state: State, *args, **kwargs):
            start_time = time.time()

            # Log state before execution
            before_state_dict = state.model_dump()
            logger.info(
                json.dumps(
                    {
                        "event": f"{node_name}_start",
                        "request_id": state.request_id,
                        "input_state": {
                            k: v for k, v in before_state_dict.items() if v
                        },
                    },
                    default=custom_json_serializer,
                )
            )

            # Execute the node
            result_state = await func(state, *args, **kwargs)

            execution_time = time.time() - start_time

            # Log state after execution
            after_state_dict = result_state.model_dump()
            logger.info(
                json.dumps(
                    {
                        "event": f"{node_name}_end",
                        "request_id": result_state.request_id,
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "output_state": {
                            k: v for k, v in after_state_dict.items() if v
                        },
                    },
                    default=custom_json_serializer,
                )
            )

            return result_state

        return wrapper

    return decorator
