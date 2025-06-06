from typing import Any
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

from .job_type import JobType
from .job_id import JobID
from .compute_platform import ComputePlatform

class AzureComputePlatform(ComputePlatform):
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        compute_name: str,
        environment_name: str,
    ):
        self._ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )
        self._compute_name = compute_name
        self._environment_name = environment_name

    def queue_job(
        self, job_type: JobType, input_name: str, input_version: str, output_name: str
    ) -> JobID:
        job = command(
            code="src/",
            command=(
                "python src/preprocess/preprocessing_job.py "
                f"--input-name {input_name} --input-version {input_version} "
                f"--output-name {output_name}"
            ),
            environment=self._environment_name + "@latest",
            compute=self._compute_name,
            experiment_name="pp_experiment",
        )
        returned = self._ml_client.jobs.create_or_update(job)
        return JobID(returned.name)

    def get_job_status(self, job_id: JobID) -> str:
        job = self._ml_client.jobs.get(str(job_id))
        return job.status

    def get_resource(self, output_name: str, version: str) -> Any:
        ds = self._ml_client.datasets.get(name=output_name, version=version)
        return ds
