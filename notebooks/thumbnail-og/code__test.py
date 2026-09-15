import sagemaker

from sagemaker.workflow.parameters import ParameterString

import datetime

class MLOpsPipelineParameters:
    def __init__(self, project_name, project_id, default_bucket):
        self.project_name = project_name
        self.default_bucket_prefix = f"{project_name}-{project_id}/Code_Location/{datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')}"
        self.model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
        self.input_data = ParameterString(name="InputData", default_value=f"s3://{default_bucket}/seed-code-data/iris.csv")
        self.bucket_kms_key_id = None
        self.pipeline_output_location = None


def create_preprocessing_resources(pipeline_sesion: sagemaker.session.Session,
                                   params: MLOpsPipelineParameters, role: str):
    
    est_cls = sagemaker.sklearn.estimator.SKLearn
    
    processing_image_uri = 
    