from dataclasses import dataclass
from typing import Optional


@dataclass
class Test:
    a: str
    b: Optional[str] = None

    # def __init__(self, a: str, b) -> None:
    #     print("GIRO")
    def __post_init__(self):
        print(self.__class__.)
        allowed_types = [type(""), type(None)]
        for field_name, field_def in self.__dataclass_fields__.items():
            v = getattr(self, field_name)
            assert (
                type(v) in allowed_types
            ), f"Error: {field_name} with type {type(v)} should have type in {allowed_types}."
            # if isinstance(field_def.type, typing._SpecialForm):
            #     # No check for typing.Any, typing.Union, typing.ClassVar (without parameters)
            #     continue


c = {"a": "5"}
d = Test(**c)
