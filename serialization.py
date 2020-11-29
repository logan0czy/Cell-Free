"""

Make an object serialized so that it can be interpreted by json.

This module refers to OpenAI spinningup.
Links: https://github.com/openai/spinningup/blob/master/spinup/utils/serialization_utils.py
"""
import json


def convertJson(obj):
    """ Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convertJson(k): convertJson(v) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convertJson(x) for x in obj)

        elif isinstance(obj, list):
            return [convertJson(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convertJson(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convertJson(k): convertJson(v) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False