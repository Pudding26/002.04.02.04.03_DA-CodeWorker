from app.utils.common.app.utils.SQL.models.jobs. api_WorkerJobs import WorkerJobs_Out
from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel

class LobbyHandler:
    """
    Handles the main loop for the TA51 tasks.
    """

    def __init__(self):
        self.jobs_to_check = ["extractor", "modeler", "validator", "creator"]

    def start(self):
        """
        Entry point to start the task-processing loop.
        """
        self.run()

    def run(self):
        """
        Runs the main loop for each TA51 task.
        """
        for job_type in self.jobs_to_check:
            orchestrator = self._get_orchestrator_for_job(job_type)
            if orchestrator is None:
                continue


            job_df_raw = self.check_for_jobs(job_type=job_type)
            if job_df_raw.empty:
                print(f"No jobs found for job type: {job_type}")
                continue

            print(f"Found {len(job_df_raw)} jobs for job type: {job_type}")
            print(f"Delegating to worker: {job_type}")

            orchestrator.run(job_df_raw=job_df_raw)

    def _get_orchestrator_for_job(self, job_type):
        """
        Factory method to return the correct orchestrator for each job type.
        """
        match job_type:
            #case "extractor":
            #    from app.tasks.TA51_Extractor.TA51_0_ExtractorOrchestrator import TA51_0_ExtractorOrchestrator
            #    return TA51_0_ExtractorOrchestrator()
            case "modeler":
                from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import TA52_0_ModelerOrchestrator
                return TA52_0_ModelerOrchestrator()
            case "validator":
                # Implement your validator orchestrator here
                return None
            case "creator":
                # Implement your creator orchestrator here
                return None
            case _:
                return None

    def check_for_jobs(self, job_type):
        """
        Fetch all jobs for the given job_type with status=in_progress.
        """
        filter_model = FilterModel.from_human_filter(
            {"contains": {"status": "READY", "job_type": job_type}}
        )
        return WorkerJobs_Out.fetch(
            filter_model=filter_model,
            stream=False
        )                                                                                                                                           
