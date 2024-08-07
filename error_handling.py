import logging
from flask import jsonify

def error_handler(f):
    def wrapper(*args, **kwargs):

        try:
            return f(*args, **kwargs)
        except Exception as e:
            #logging.error(f"an error is occured")
            return f"error', 'internal error occured: {e}", 500
    return wrapper
    