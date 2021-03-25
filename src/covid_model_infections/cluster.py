from bdb import BdbQuit
from collections import Counter
import os
from pathlib import Path
import shutil
import time
from typing import Dict, List

from covid_shared import shell_tools
from loguru import logger


PROJECT = 'proj_covid_prod'
QUEUE = 'all.q'
F_MEM = '5.0G'
F_THREAD = '6'
OMP_NUM_THREADS = '6'
H_RUNTIME = '01:00:00'
SLEEP_TIME = 10


def run_cluster_jobs(job_type: str, output_root: Path, job_args_map: Dict[int, List[str]]) -> None:
    drmaa = get_drmaa()
    jobs = {}
    with drmaa.Session() as session:
        try:
            logger.info(f"Enqueuing {job_type} jobs...")
            for job_id, job_args in job_args_map.items():
                job_name = f'{job_type}_{job_id}'
                job = do_qsub(session, job_type, job_name, output_root, job_args)
                jobs[job_name] = (job, drmaa.JobState.UNDETERMINED)

            logger.info('Entering monitoring loop.')
            logger.info('-------------------------')
            logger.info('')

            while any([job[1] not in [drmaa.JobState.DONE, drmaa.JobState.FAILED] for job in jobs.values()]):

                statuses = Counter()
                for job_name, (job_id, status) in jobs.items():
                    new_status = session.jobStatus(job_id)
                    jobs[job_name] = (job_id, new_status)
                    statuses[new_status] += 1
                for status, count in statuses.items():
                    logger.info(f'{status:<35}: {count:>4}')
                logger.info('')
                time.sleep(SLEEP_TIME)
                logger.info('Checking status again')
                logger.info('---------------------')
                logger.info('')
        except (KeyboardInterrupt, BdbQuit) as e:
            logger.info('User termination of the master process. Killing all running jobs.')
            for job_name, (job_id, status) in jobs.items():
                if status not in [drmaa.JobState.DONE, drmaa.JobState.FAILED]:
                    session.control(job_id, drmaa.JobControlAction.TERMINATE)
            raise e

    logger.info('**Done**')


def do_qsub(session, job_type: str, job_name: str, output_root: Path, script_args: List[str]):
    error_logs = output_root / 'logs' / job_type / 'error'
    output_logs = output_root / 'logs' / job_type / 'output'
    shell_tools.mkdir(error_logs, exists_ok=True, parents=True)
    shell_tools.mkdir(output_logs, exists_ok=True, parents=True)

    job_template = session.createJobTemplate()
    job_template.remoteCommand = shutil.which('python')
    job_template.outputPath = f':{output_logs}'
    job_template.errorPath = f':{error_logs}'
    job_template.args = script_args
    job_template.nativeSpecification = (f'-V '  # Export all environment variables
                                        f'-b y '  # Command is a binary (python)
                                        f'-P {PROJECT} '
                                        f'-q {QUEUE} '
                                        f'-l fmem={F_MEM} '
                                        f'-l fthread={F_THREAD} '
                                        f'-l h_rt={H_RUNTIME} '
                                        f'-N {job_name}')  # Name of the job
    job = session.runJob(job_template)
    logger.info(f'Submitted job {job_name} with id {job}.')
    session.deleteJobTemplate(job_template)
    return job


def decode_status(job_status):
    """Decodes a UGE job status into a string for logging"""
    drmaa = get_drmaa()
    decoder_map = {drmaa.JobState.UNDETERMINED: 'undetermined',
                   drmaa.JobState.QUEUED_ACTIVE: 'queued_active',
                   drmaa.JobState.SYSTEM_ON_HOLD: 'system_hold',
                   drmaa.JobState.USER_ON_HOLD: 'user_hold',
                   drmaa.JobState.USER_SYSTEM_ON_HOLD: 'user_system_hold',
                   drmaa.JobState.RUNNING: 'running',
                   drmaa.JobState.SYSTEM_SUSPENDED: 'system_suspended',
                   drmaa.JobState.USER_SUSPENDED: 'user_suspended',
                   drmaa.JobState.DONE: 'finished',
                   drmaa.JobState.FAILED: 'failed'}

    return decoder_map[job_status]


def get_drmaa():
    try:
        os.environ['DRMAA_LIBRARY_PATH'] = '/opt/sge/lib/lx-amd64/libdrmaa.so'
        import drmaa
    except (RuntimeError, OSError):
        raise Exception("drmaa could not be imported!")
    return drmaa
