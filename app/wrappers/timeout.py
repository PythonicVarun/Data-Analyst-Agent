import signal
import inspect
import asyncio
import threading
from functools import wraps
from concurrent import futures


class TimeoutError(Exception):
    """Custom exception raised when a function times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler that raises our custom exception."""
    raise TimeoutError("Function call timed out!")


def _parse_timeout_value(value):
    """Return int timeout in seconds or None if not provided/invalid."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        try:
            fv = float(value)
            return int(fv) if fv > 0 else None
        except Exception:
            return None

    try:
        val = float(value)
        if val > 0:
            return int(val)
    except Exception:
        pass
    return None


def timeout(_func=None, *, timeout=None):
    """
    Decorator factory that optionally accepts a decorator-level timeout.

    Usage:
        @timeout                 # no default; use per-call `timeout=` or no timeout
        def f(..., timeout=...): ...

        @timeout(timeout=5)      # default timeout of 5s; decorator timeout takes priority
        def f(...): ...

        @timeout
        async def af(...): ...

        @timeout(timeout=3)
        async def af(...): ...

    Priority: decorator-level `timeout=` (if provided and valid) > per-call `timeout=` kwarg > no timeout.
    """
    decorator_timeout = _parse_timeout_value(timeout)

    def _decorate(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                raw_call_timeout = kwargs.pop("timeout", None)
                call_timeout = _parse_timeout_value(raw_call_timeout)
                t = decorator_timeout if decorator_timeout is not None else call_timeout

                try:
                    if t:
                        return await asyncio.wait_for(func(*args, **kwargs), timeout=t)
                    else:
                        return await func(*args, **kwargs)
                except asyncio.TimeoutError:
                    print(
                        f"Error: Function '{func.__name__}' timed out after {t} seconds."
                    )
                    return None

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                raw_call_timeout = kwargs.pop("timeout", None)
                call_timeout = _parse_timeout_value(raw_call_timeout)
                t = decorator_timeout if decorator_timeout is not None else call_timeout

                can_use_sigalrm = (
                    hasattr(signal, "SIGALRM")
                    and threading.current_thread() is threading.main_thread()
                )

                if t and can_use_sigalrm:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(t)
                    try:
                        return func(*args, **kwargs)
                    except TimeoutError:
                        print(
                            f"Error: Function '{func.__name__}' timed out after {t} seconds."
                        )
                        return None
                    finally:
                        signal.alarm(0)
                elif t:
                    with futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        try:
                            return future.result(timeout=t)
                        except futures.TimeoutError:
                            print(
                                f"Error: Function '{func.__name__}' timed out after {t} seconds."
                            )
                            return None
                else:
                    # No timeout specified; call directly
                    return func(*args, **kwargs)

            return sync_wrapper

    # handle decorator used with or without parentheses:
    # - @timeout  -> _func is the function
    # - @timeout(timeout=3) -> _func is None, returns decorator
    if _func is None:
        return _decorate
    else:
        return _decorate(_func)
