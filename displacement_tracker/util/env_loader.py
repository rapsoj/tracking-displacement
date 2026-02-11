import os
from functools import wraps
from dotenv import load_dotenv


class EnvFileNotFoundError(FileNotFoundError):
    pass


class EnvKeyMissingError(KeyError):
    pass


def require_env_file(required_keys=None):
    """
    Decorator to ensure a .env file exists and contains required keys before running the function.
    Usage:
        @require_env_file
        def foo(...): ...

        @require_env_file(['KEY1', 'KEY2'])
        def bar(...): ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Locate .env file in current or parent directories
            current_dir = os.getcwd()
            env_path = None
            while True:
                candidate = os.path.join(current_dir, ".env")
                if os.path.isfile(candidate):
                    env_path = candidate
                    break
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    raise EnvFileNotFoundError(
                        ".env file not found in current or parent directories."
                    )
                current_dir = parent_dir
            # Parse .env file
            load_dotenv(env_path)
            # Validate required keys
            if required_keys:
                missing = [k for k in required_keys if not os.getenv(k)]
                if missing:
                    raise EnvKeyMissingError(
                        f"Missing required keys in env: {', '.join(missing)}"
                    )
            return func(*args, **kwargs)

        return wrapper

    # Support both @require_env_file and @require_env_file([...])
    if callable(required_keys):
        return decorator(required_keys)
    return decorator
