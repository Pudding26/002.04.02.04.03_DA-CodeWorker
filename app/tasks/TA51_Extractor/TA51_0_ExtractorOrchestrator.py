from app.utils.common.app.utils.dataModels.Jobs.ExtractorJob import ExtractorJob, ExtractorJobInput
from app.utils.common.app.tasks.TA41_ImageSegmentation.FeatureExtractor import FeatureExtractor



class TA51_0_ExtractorOrchestrator:
    """
    This class orchestrates the extraction process for TA51 data.
    It initializes the extractor and manages the extraction workflow.
    """

    def __init__(self):
        """
        Initializes the TA51_0_ExtractorOrchestrator with a given extractor.

        :param extractor: An instance of the extractor to be used for data extraction.
        """
        pass

    def run(self, job_df_raw):
        """
        Runs the extraction process using the initialized extractor.
        """
        self.jobs = self.create_job_list(job_df_raw=job_df_raw)

        self.run_pipeline()

    



    def create_job_list(self, job_df_raw):

        """
        Converts a DataFrame of raw job data into a list of ExtractorJob instances.

        :param job_df_raw: DataFrame containing raw job data.
        :return: List of ExtractorJob instances.
        """
        jobs = []
        for index, row in job_df_raw.iterrows():
            payload = row["payload"]["input"]

            job_input = ExtractorJobInput(
                mask=payload.get("mask"),           # can be None
                n_images=payload["n_images"],
                width=payload["width"],
                height=payload["height"],
                stackID=payload["stackID"],
            )

            job_input.mask = job_input.get_mask()

            job = ExtractorJob(
                job_uuid=row['job_uuid'],
                job_type=row['job_type'],
                status=row['status'],
                attempts=row['attempts'],
                next_retry=row['next_retry'],
                created=row['created'],
                updated=row['updated'],
                parent_job_uuids=row['parent_job_uuids'],  # list[str] per your schema
                input=job_input
            )

            jobs.append(job)

        return jobs
    
    def run_pipeline(self):

        fe = FeatureExtractor() 

        for job in self.jobs:

            mask_stack = job.input.mask
            for mask in mask_stack:
                # Apply the feature extractor to each mask
                feature_df = fe.apply_one(mask, connectivity = 2, use_gpu=True)


