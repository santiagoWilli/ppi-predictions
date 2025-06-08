from abc import ABC, abstractmethod

class ProteinEncoder(ABC):
    @abstractmethod
    def encode(self, sequence: str) -> list:
        pass

    @abstractmethod
    def vocabulary_size(self) -> int:
        pass
