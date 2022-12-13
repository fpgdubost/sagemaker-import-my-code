import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
)

print(sagemaker.__version__)

sess = sagemaker.Session()  # Use the AWS region configured with the AWS CLI

role = 'arn:aws:iam::931683541123:role/service-role/AmazonSageMaker-ExecutionRole-20220610T135217'

estimator = Estimator(  
    entry_point='train.py', # needs to be in the local path, but does not actually read it, nto sure why, but it should be called train 
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge', # fast launch: ml.g4dn.xlarge ml.g5.4xlarge
    image_uri="931683541123.dkr.ecr.us-west-2.amazonaws.com/training-example11:latest",
    base_job_name='training-job',
    output_path='s3://sagemaker-training-example/output_sagemaker/'
)

hyperparameter_ranges = {
    "lr": ContinuousParameter(1e-7, 1e-3),
    "aug_shift": ContinuousParameter(0., 0.3),
    "aug_rot": ContinuousParameter(0., 0.4),
}

objective_metric_name = "val_loss"
objective_type = "Minimize"
metric_definitions = [{'Name': 'val_loss', 'Regex': 'val_loss: ([0-9\\.]+)'}]

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    objective_type=objective_type,
    max_jobs=9,
    max_parallel_jobs=3,
)

# Train! This will pull (once) the SageMaker CPU/GPU container for TensorFlow to your local machine.
# Make sure that Docker is running and that docker-compose is installed

tuner.fit('s3://sagemaker-training-example/data/2', wait=False)




