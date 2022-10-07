from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable

class IndependenceTest(ABC):
    # def __init__(self) -> None:
    #     pass
    @abstractmethod
    def run(self,
            x:str,
            y:str,
            s:Iterable[str],
            data:dict,
            verbose=None,
            **kwargs # Specific test parameters
            ) -> IndependenceTest.TestResults:
        pass
    class TestResults():
        @abstractmethod
        def independent(self):
            pass
