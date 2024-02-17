import functools
import random
import time
import typing as t

from tg.utils.logger import logger


def exponential_backoff(
        max_iterations: int = 32, exponent_limit: int = 5
) -> t.Callable:
    """
    Creates a decorator that implements exponential backoff for inner function.
    """

    def decorator(func: t.Callable) -> t.Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> t.Any:
            """
            Returns delegated `func` result if the run is successful. Beware that this may change the signature of
            `func`. If the decorated function previously couldn't return None, after using this decorator it can.
            """
            for i in range(max_iterations):
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    msg = str(ex)
                    logger.info(f"Exception {msg} occurred during inner call.")
                    # We were not successful in any of the attempts, just raise inner ex
                    if i + 1 == max_iterations:
                        raise ex
                exponent = min(i, exponent_limit)
                secs = (2 ** exponent) + random.random()
                logger.info(
                    f"Exponential backoff iteration: {i} | Sleep time: {secs} s"
                )
                time.sleep(secs)
            # This part of the code should be mathematically impossible to reach
            raise Exception("Unreachable code reached")

        return wrapper

    return decorator


def exponential_backoff_async(
        max_iterations: int = 32, exponent_limit: int = 5
) -> t.Callable:
    """
    Creates a decorator that implements exponential backoff for inner function in async variant.
    """

    def decorator(func: t.Callable) -> t.Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> t.Any:
            """Returns delegated `func` result if the run is successful.
            Beware that this may change the signature of `func`. If the decorated function previously couldn't
            return None, after using this decorator it can."""
            for i in range(max_iterations):
                try:
                    return await func(*args, **kwargs)
                except Exception as ex:
                    msg = str(ex)
                    logger.info(f"Exception {msg} occurred during inner call.")
                    # We were not successful in any of the attempts, just raise inner ex
                    if i + 1 == max_iterations:
                        raise ex
                exponent = min(i, exponent_limit)
                secs = (2 ** exponent) + random.random()
                logger.info(
                    f"Exponential backoff iteration: {i} | Sleep time: {secs} s"
                )
                time.sleep(secs)
            # This part of the code should be mathematically impossible to reach
            raise Exception("Unreachable code reached")

        return wrapper

    return decorator
