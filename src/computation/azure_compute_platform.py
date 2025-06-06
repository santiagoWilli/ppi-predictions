from pathlib import Path
from typing import Any
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
from .job_type import JobType
from .job_id import JobID
from .compute_platform import ComputePlatform

class AzureComputePlatform(ComputePlatform):
    def __init__(
        self,
        subscription_id: str  = None,
        resource_group: str   = None,
        workspace_name: str   = None,
        compute_name: str     = None,
        environment_name: str = None,
    ):
        if subscription_id and resource_group and workspace_name and compute_name and environment_name:
            self._ml_client = MLClient(
                DefaultAzureCredential(), subscription_id, resource_group, workspace_name
            )
            self._compute_name = compute_name
            self._environment_name = environment_name
        else:
            # For AML jobs
            self._ml_client = MLClient(DefaultAzureCredential())

    def queue_job(
        self, job_type: JobType, input_name: str, input_version: str, output_name: str
    ) -> JobID:
        src_path = str(Path(__file__).resolve().parents[1])
        job = command(
            code=src_path,
            command=(
                "python computation/jobs/preprocessing_job.py "
                "--input-path ${{inputs.ppi_input}} "
                "--output-dir ${{outputs.ppi_output}}"
            ),
            environment=self._environment_name,
            compute=self._compute_name,
            experiment_name="ppi_experiment",
            inputs={
                "ppi_input": Input(
                    type=AssetTypes.URI_FOLDER, 
                    path=f"{input_name}:{input_version}",
                    mode=InputOutputModes.RO_MOUNT
                )
            },
            outputs={
                "ppi_output": Output(
                    type=AssetTypes.URI_FOLDER, 
                    mode=InputOutputModes.RW_MOUNT
                )
            },
        )
        returned = self._ml_client.jobs.create_or_update(job)
        return JobID(returned.name)

    def get_job_status(self, job_id: JobID) -> str:
        job = self._ml_client.jobs.get(str(job_id))
        return job.status

    def get_resource(self, name: str, version: str) -> Any:
        return self._ml_client.data.get(name=name, version=version)
