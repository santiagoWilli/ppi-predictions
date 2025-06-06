from abc import ABC, abstractmethod
from typing import Any
from .job_id import JobID
from .job_type import JobType

class ComputePlatform(ABC):
    @abstractmethod
    def queue_job(
        self, job_type: JobType, input_name: str, input_version: str, output_name: str
    ) -> JobID:
        pass

    @abstractmethod
    def get_job_status(self, job_id: JobID) -> str:
        pass

    @abstractmethod
    def get_resource(self, name: str, version: str) -> Any:
        pass
