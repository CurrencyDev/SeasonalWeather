"""Repository modeling primitives shared without importing API ownership."""

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator, model_validator

__all__ = [
    "BaseModel",
    "ConfigDict",
    "Field",
    "SerializeAsAny",
    "field_validator",
    "model_validator",
]
