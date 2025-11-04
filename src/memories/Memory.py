from typing import Optional, List, Any
from dataclasses import dataclass, asdict
from dataloaders.ProblemState import ProblemState


class MemoryClassMeta(type):
    def __new__(cls, name, bases, namespace, field_names=None):
        # Inject type annotations for dataclass to use
        if field_names:
            annotations = {}
            for field in field_names:
                annotations[field] = Any  # or a specific type
            namespace['__annotations__'] = annotations
        clsobj = super().__new__(cls, name, bases, namespace)
        return dataclass(clsobj)  # Apply @dataclass dynamically
    
class BaseMemory(metaclass=MemoryClassMeta, field_names=["ps"]):
    pass


class ReflexionMemory(metaclass=MemoryClassMeta, field_names=["ps", "err_msg", "reflection"]):
    pass


# class FunctionSignatureMemory(BaseMemory):
#     function_signatures: Optional[List] = None


# @dataclass
# class OneshotMemory(BaseMemory):
#     oneshot: Optional[str] = None

