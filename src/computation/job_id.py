class JobID:
    def __init__(self, id_str: str):
        self._id = id_str

    def __str__(self) -> str:
        return self._id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JobID):
            return NotImplemented
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)
