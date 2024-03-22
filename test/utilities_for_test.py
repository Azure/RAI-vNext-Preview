# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

from azure.ai.ml import MLClient
from azure.ai.ml.entities import PipelineJob

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def submit_and_wait(
    ml_client: MLClient, pipeline_job: PipelineJob, expected_state: str = "Completed"
) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None
    info_msg = f"Created job: {created_job}"
    print(info_msg)
    _logger.info(info_msg)
    try:
        ml_client.jobs.stream(created_job.name)
    except Exception as e:
        _logger.error(f"Error while streaming job: {e}")
    # get latest status once job finished
    created_job = ml_client.jobs.get(created_job.name)
    if created_job.status != expected_state:
        _logger.error(str(created_job))
    assert created_job.status == expected_state
    return created_job


def process_file(input_file, output_file, replacements):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            for f, r in replacements.items():
                line = line.replace(f, r)
            outfile.write(line)
