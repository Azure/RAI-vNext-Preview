# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import time

from azure.ml import MLClient, PipelineJob


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def submit_and_wait(
    ml_client: MLClient, pipeline_job: PipelineJob, expected_state: str = "Completed"
) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    terminal_states = ["Completed", "Failed", "Canceled", "NotResponding"]
    assert created_job is not None
    assert expected_state in terminal_states

    while created_job.status not in terminal_states:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print("Latest status : {0}".format(created_job.status))
        _logger.info("Latest status : {0}".format(created_job.status))
    if created_job.status != expected_state:
        _logger.error(str(created_job))
    assert created_job.status == expected_state
    return created_job
