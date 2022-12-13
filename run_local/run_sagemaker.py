import sagemaker
from sagemaker.estimator import Estimator

print(sagemaker.__version__)

sess = sagemaker.Session()  # Use the AWS region configured with the AWS CLI
# sess = sagemaker.Session(boto3.session.Session(region_name='eu-west-1'))

# This doesn't work on your local machine because it doesn't have an IAM role :)
# role = sagemaker.get_execution_role()

# This is the SageMaker role you're already using, it will work just fine
role = 'arn:aws:iam::931683541123:role/service-role/AmazonSageMaker-ExecutionRole-20220610T135217'

# Store model locally. A S3 URI would work too.
output_path = 'file:///home/ubuntu/sagemaker-setup-example/test_local_outputs/1/'

estimator = Estimator(  
    entry_point='train.py', # needs to be in the local path, but does not actually read it, not sure why, but it should be called train 
    role=role,
    instance_count=1,
    instance_type='local',
    image_uri="931683541123.dkr.ecr.us-west-2.amazonaws.com/training-example11:latest",
    output_path=output_path
)

estimator.fit('file:////home/ubuntu/sagemaker-setup-example/data/2/')




