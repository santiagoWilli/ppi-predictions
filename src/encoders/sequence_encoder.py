from abc import ABC, abstractmethod

class SequenceEncoder(ABC):
    @abstractmethod
    def encode(self, sequence: str) -> list:
        pass

    @abstractmethod
    def decode(self, encoded: list) -> str:
        pass

    @abstractmethod
    def vocabulary_size(self) -> int:
        pass
