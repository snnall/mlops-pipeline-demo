import os
import time
import boto3
import sagemaker
import json
import sys

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput,CreateModelInput,TransformInput

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
)

from sagemaker import Model
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.huggingface import HuggingFace, HuggingFaceModel

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig,TuningStep,TransformStep
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.workflow.step_collections import CreateModelStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    #JsonGet,
)
from sagemaker.workflow.functions import (JsonGet)
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.transformer import Transformer


region = sagemaker.Session().boto_region_name
sm_client = boto3.client("sagemaker")
s3_client = boto3.client('s3', region_name=region)
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session, sagemaker_client=sm_client
)
s3_prefix = "output"
model_package_group_name = f"RegMLNBModelPackageGroupName"

# Opening JSON file with CDK outputs
f = open('./cdk-outputs.json')

data = json.load(f)
f.close()

# Pull role arn key from the json list of CDK outputs
regml_output = data['regml-stack']
role_arn_key = list(regml_output.keys())[0]
role = data['regml-stack'][role_arn_key]

print('Role:', role)
lambda_role = role
default_bucket = sagemaker_session.default_bucket()
output_destination = "s3://{}/{}".format(default_bucket, s3_prefix)

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

# processing step parameters
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
processing_instance_type = ParameterString(
    name="ProcessingInstanceType", default_value="ml.m5.xlarge"
)

# training step parameters
training_instance_type = ParameterString(
    name="TrainingInstanceType", default_value="ml.m5.xlarge"
)
training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)

# model approval status
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
)

# input data
input_data = ParameterString(
    name="InputData",
    default_value="s3://{}/data/storedata_total.csv".format(default_bucket),  # Change this to point to the s3 location of your raw input data.
    )
    
# batch data    
batch_data = ParameterString(
    name="BatchData",
    default_value="s3://{}/data/batch/batch.csv".format(default_bucket),  # Change this to point to the s3 location of your raw input data.
    )
    
# cache configuration
cache_config = CacheConfig(enable_caching=True, expire_after="30d")

########## Data Processing ##########
print('Data Processing Step ..')
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="RegMLNB-preprocessing",
    role=role,
)

step_process = ProcessingStep(
    name="Processing",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),  
        ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            destination="{}/train".format(output_destination),
            source="/opt/ml/processing/train",
        ),
        ProcessingOutput(
            output_name="test",
            destination="{}/test".format(output_destination),
            source="/opt/ml/processing/test",
        ),
        ProcessingOutput(
            output_name="validation",
            destination="{}/validation".format(output_destination),
            source="/opt/ml/processing/validation",
        ),
    ],
    code="./RegMLNB/preprocessing_xg.py",
    #job_arguments=["--input-data", input_data],
    cache_config=cache_config,
)

########## Training ##########

print('Training Step ..')
model_path = f"s3://{default_bucket}/output"
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.0-1",
    py_version="py3",
    instance_type=training_instance_type,
)
fixed_hyperparameters = {
"eval_metric":"auc",
"objective":"binary:logistic",
"num_round":"100",
"rate_drop":"0.3",
"tweedie_variance_power":"1.4"
}
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type=training_instance_type,
    instance_count=1,
    hyperparameters=fixed_hyperparameters,
    output_path=model_path,
    base_job_name=f"train",
    sagemaker_session=sagemaker_session,
    role=role,
)
hyperparameter_ranges = {
    "eta": ContinuousParameter(0, 1),
    "min_child_weight": ContinuousParameter(1, 10),
    "alpha": ContinuousParameter(0, 2),
    "max_depth": IntegerParameter(1, 10),
}

objective_metric_name = "validation:auc"

step_tuning = TuningStep(
name = "ModelTrainingTuning",
tuner = HyperparameterTuner(xgb_train, objective_metric_name, hyperparameter_ranges, max_jobs=2, max_parallel_jobs=2),
inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
)


########## Model Evaluation ##########

print('Evaluation Step ..')

# Processing step for evaluation
script_eval = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name=f"script-RegMLNB-eval",
    sagemaker_session=sagemaker_session,
    role=role,
)

evaluation_report = PropertyFile(
    name="valuationReport",
    output_name="evaluation",
    path="evaluation.json",
)
step_eval = ProcessingStep(
    name="ModelEvaluation",
    processor=script_eval,
    inputs=[
        ProcessingInput(
            source=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=default_bucket,prefix="output"),
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        # ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination = f"{output_destination}/evaluation_report",
        #                 ),
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",\
                         destination=f"s3://{default_bucket}/output/evaluation")
    ],
    code="./RegMLNB/evaluate_xg.py",
    property_files=[evaluation_report],
    cache_config=cache_config,
)

########## MODEL REGISTRATION AND APPROVAL STEP ##########
print('Model registration Step ..')

model = Model(
    image_uri=image_uri,        
    model_data=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=default_bucket,prefix="output"),
    sagemaker_session=sagemaker_session,
    role=role,
)
inputs = CreateModelInput(
    instance_type="ml.m5.large",
    accelerator_type="ml.eia1.medium",
)
step_create_model = CreateModelStep(
    name="CreateModel",
    model=model,
    inputs=inputs,
)

# step to perform batch transformation
transformer = Transformer(
model_name=step_create_model.properties.ModelName,
instance_type="ml.m5.xlarge",
instance_count=1,
output_path=f"s3://{default_bucket}/BatchPredictions"
)
step_transform = TransformStep(
name="BatchPredictions",
transformer=transformer,
inputs=TransformInput(data=batch_data,content_type="text/csv")
)

from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json",
    )
)
step_register = RegisterModel(
    name="RegisterModel",
    estimator=xgb_train,
    model_data=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=default_bucket,prefix="output"),
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    # model=model,
    # content_types=["application/json"],
    # response_types=["application/json"],
    # inference_instances=["ml.g4dn.xlarge", "ml.m5.xlarge"],
    # transform_instances=["ml.g4dn.xlarge", "ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status= 'Approved', #model_approval_status
    model_metrics=model_metrics,
)

########## LAMBDA STEP FOR ENDPOINT CREATION ##########
print('Lambda Step ..')

current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
model_name = "RegMLNB-model" + current_time
endpoint_config_name = "RegMLNB-endpoint-config" + current_time
endpoint_name = "RegMLNB-endpoint-" + current_time
function_name = "RegMLNB-lambda-step" + current_time

# Lambda helper class can be used to create the Lambda function
func = Lambda(
    function_name=function_name,
    execution_role_arn=lambda_role,
    script="lambda_deployer.py",
    handler="lambda_deployer.lambda_handler",
    timeout=600,
    memory_size=3008, #10240
)

# The dictionary retured by the Lambda function is captured by LambdaOutput, each key in the dictionary corresponds to a
# LambdaOutput

output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)
output_param_3 = LambdaOutput(output_name="other_key", output_type=LambdaOutputTypeEnum.String)

# The inputs provided to the Lambda function can be retrieved via the `event` object within the `lambda_handler` function
# in the Lambda
step_deploy_lambda = LambdaStep(
    name="LambdaStepRegMLNBDeploy",
    lambda_func=func,
    inputs={
        "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "endpoint_name": endpoint_name,
        "model_package_arn": step_register.steps[0].properties.ModelPackageArn,
        "role": role,
        "data_capture_destination": "{}/datacapture".format(output_destination)
    },
    outputs=[output_param_1, output_param_2, output_param_3],
)


########## MODEL QUALITY MONITOR STEP ##########
print("Model Quality Monitor step...")

# Model Quality Monitor
from sagemaker import get_execution_role, session, Session
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from datetime import datetime

# SKLearnProcessor to run model quality monitor job
model_quality_processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    volume_size_in_gb=60,
    base_job_name='model-quality',
    sagemaker_session=sagemaker_session)

# Upload fake_train_data.csv to fit baselining job
s3_client.upload_file(Filename='./data/fake_train_data.csv', Bucket=default_bucket, Key=f'{s3_prefix}/model-monitor/fake_train_data.csv')
baseline_dataset_uri = f's3://{default_bucket}/{s3_prefix}/model-monitor/fake_train_data.csv'

ground_truth_upload_path = (
    f"s3://{default_bucket}/{s3_prefix}/ground-truth-data"
)

# Processing step to execute batch inference script in processing container
step_model_quality = ProcessingStep(
    name='ModelQualityScheduleStep',
    processor=model_quality_processor,
    job_arguments=[
        "--endpoint-name", 'test-endpoint',
        "--baseline-dataset-uri", baseline_dataset_uri,
        "--ground-truth-upload-path", ground_truth_upload_path,
        "--role", role,
        "--region", region,
    ],
    code="./RegMLNB/monitor.py",
    #depends_on=[step_deploy_lambda]
    )


########## CONDITION STEP TO CHECK MODEL ACCURACY AND CONDITIONALLY CREATE A MODEL AND REGISTER IN MODEL REGISTRY ##########

print('Conditional Step ..')

# Condition step for evaluating model accuracy and branching execution
cond_accuracy = ConditionGreaterThanOrEqualTo(  # You can change the condition here
    left=JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="classification_metrics.auc_score.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.

    ),
    right=0.60,  # You can change the threshold here
)

step_cond = ConditionStep(
    name="CheckEvalAccuracy",
    conditions=[cond_accuracy ],#cond_prauc
    if_steps=[step_register,step_create_model,step_transform], #step_batch_transform,step_lambda_invoke,step_deploy_lambda
    #if_steps=[step_register, step_create_model,step_transform],
    else_steps=[step_model_quality],
)

########## DEFINE PIPELINE ##########

try: 
    pipeline_name = f"DemoMLOpsPipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            model_approval_status,
            input_data,
            batch_data
        ],
        steps=[step_process,step_tuning,step_eval,step_cond], # 
        sagemaker_session=sagemaker_session,
    )

    definition = json.loads(pipeline.definition())

    #submit the pipeline to sagemaker and start execution
    upsert_response = pipeline.upsert(role_arn=role) 
    print("\n###### Created/Updated SageMaker Pipeline: Response received:")
    print(upsert_response)

    execution = pipeline.start()
    print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

    print("Waiting for the execution to finish...")
    execution.wait() # wait
    print("\n#####Execution completed. Execution step details:")
    print(execution.list_steps())

    
except Exception as e:  
    print(f"Exception: {e}")
    sys.exit(1)



