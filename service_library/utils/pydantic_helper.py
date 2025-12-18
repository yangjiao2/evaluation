from pydantic import BaseModel


def is_valid_instance(attribute):
    """
    Checks if the given attribute is a valid Pydantic model instance.

    Returns True if it's an instance of BaseModel, otherwise False.
    """
    return isinstance(attribute, BaseModel) and not isinstance(attribute, type)
