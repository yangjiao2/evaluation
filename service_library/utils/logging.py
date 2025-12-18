import logging
import os
from functools import wraps

def log_errors(action):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(os.path.basename(__file__))
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                logger.error('Error in function %s when %s: %s', func.__name__, action, ex)
                print('🧨Execution Error in function %s when %s: %s', func.__name__, action, ex)
        return wrapper
    return decorator