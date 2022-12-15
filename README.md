# Sagemaker Training Example

This repository provides a step-by-step tutorial for integrating your own machine learning code, pipeline, and data with Amazon SageMaker. The tutorial includes example code based on CIFAR experiments and is accompanied by a text and [video guide](https://www.youtube.com/watch?v=XIIJnBh8sb4&t=573s&ab_channel=FlorianDubost). The tutorial covers installing SageMaker, installing AWS CLI, installing Docker, extending an existing Docker image for TensorFlow with your libraries, adding your own code to the image in the SageMaker directory format, running your model locally with SageMaker, running your model on the cloud with SageMaker, and running hyperparameter optimization on the cloud with Bayesian search. The tutorial is intended to be a valuable resource for machine learning practitioners looking to use SageMaker and scale up their experiments at minimal cost for their organization. TensorFlow is showcased here, but using other ML libraries should work too.


You don&#39;t need to run anything from SageMaker (SM) studio or Notebooks. I recommend running everything from your own computer or from an EC2 instance as it makes everything simpler. You could also just run this on your own desktop or laptop. If you use SageMaker Studio or Notebooks with the instructions below, it may lead to errors. Please run the step below in order. You need to have docker, docker compose and sagemaker installed on your local machine, but no worries, we will cover those.

## 0. Set up an EC2 instance (optional)

This is an optional step as you could run all the steps below on your own machine. But for the sake of completeness, I cover it in the video. 
Most important information here is that I tested the code with Ubuntu 22 and that worked fine.


## 1. Install and setup AWS CLI

This is not require for that step, but you will need to do it at some point you may as well do it now: install SageMaker python package locally.
```
pip3 install sagemaker
```

Then you would install the AWS CLI, as we will use it to set up the containers on AWS ECR:

https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html

[https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)

### Change your default region

It is saved in `~/.aws/config`

The file should be:
```
[default]
region=us-west-2
```

Or

pass/override it with the CLI call:

`aws ecs list-container-instances --cluster default --region us-west-2`

## 2. Clear Docker environment

Follow the instructions here: https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository

Docker can be tricky to install and setup correctly. Follow the steps below if you have some troubles.

[Install docker](https://docs.docker.com/desktop/install/ubuntu/) if not already on your system.

Follow [this](https://askubuntu.com/questions/1409192/cannot-install-docker-desktop-on-ubuntu-22-04) in you have problems installing docker. 

You may need to install docker-compose as well.
```
sudo apt install docker-compose
```

Test docker with the follwoing:
```
docker run hello-world
```
If you get the following error message
```
Got permission denied while trying to connect to the Docker daemon socket.
```
Run the following
```
sudo groupadd docker
sudo usermod -aG docker ${USER}
```
Log out and in again.

If you already have docker installed and have conflicts with existing containers, you may need to do the following:
```
sudo systemctl stop docker

sudo systemctl daemon-reload

sudo systemctl start docker

docker network prune
```

## 3. Extend official SM TF container

This step can take a while because of the installation and pushing. For some reason, it is much faser if I do it on the EC2 instance. It takes forever on my laptop. If you are not using an EC2 instance, you will have to do this step only once, so don't worry too much about it. Changes in code will be adressed in step 4 and are much faster to process.

Check out the list of (official AWS containers)[https://github.com/aws/deep-learning-containers/blob/master/available\_images.md](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)

You can chose which one you want to extend from and write the name in the Dockerfile, first line (FROM). If you don't find your desired TensorFlow in the list above, you can always add it to `extend_tf2.8/container/requirements.txt`.

Go to folder `extend_tf2.8`.

Change `algorithm_name` variable line 4 in `create_container.sh`.

Run `create_container.sh`.

### Optional:

Test container interactively, replace `763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.8.0-gpu-py39-cu112-ubuntu20.04-sagemaker` with your container ID, which you can find on ECR on the AWS web platform and run the following:

`docker run -it --rm --runtime=nvidia 931683541123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-extend`

or

`docker run -it --rm 931683541123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-extend`

## 4. Add code to container

I created another container because I find it faster to debug. That way, when you want to change your code, you don&#39;t have to keep running the Dockerfile that installs all your dependencies over and over again.

You entrypoint script should be named `train.py`.

You should not have a file called `requirements.txt` in the same folder as your entrypoint (`train.py`). Otherwise SM will try to install the list listed packages (and probably fail and exit the job).

Go to folder `adding_code`.

Add you code in the folder `container/code`.

Adapt the Dockerfile `adding_code/container/Dockerfile` line 4 such that you have your entrypoint `train.py` at location `/opt/ml/code` in your container.

Adapt the name of the container at the first line of the Dockerfile using the name defined in the section above.

Change `algorithm_name` variable line 4 in `create_container.sh`.

Run `create_container.sh`.

### Environment variable to communicate with sagemaker path

All SM env variables are stored in `SM_TRAINING_ENV`. This is automatically printed in the terminal at the beginning when running `estimator.fit`. This function is called in `run_local/run_sagemaker.py`, a python script written for Sagemaker (see next section, step 5).

#### DATA IN

If you pass a string to `estimator.fit` in the next section, then the path you want to access within the python script for your data is stored in `os.environ.get('SM_CHANNEL_TRAINING')`.

(You can also pass a dict with `{channel_name, path}` to organize your folder into subfolder. But my application, that is not necessary.)

#### DATA OUT

You can save all your outputs and logs to the path given by `os.environ.get('SM_OUTPUT_DATA_DIR')`. You need to give a path (local or s3) as argument `output_path` of the estimator during declaration in the next section.

### TIPS
I recommend passing a directory path to `estimator.fit`. Your data should be in that directory. SageMaker will copy the content of that directory to `/opt/ml/input/train` in the container before starting your python script. 


For example, we pass `'file:////home/ubuntu/sagemaker-setup-example/data/2/'` to `estimator.fit`. This is a local directory that contains our dataset file `'dataset_for_training_risk_level_5.h5'`. During launch, SageMaker will copy `'dataset_for_training_risk_level_5.h5'` to `/opt/ml/input/train`.
In `adding_code/container/code/cifar/network/segmentation_basis/train.py`, we define the path to our dataset as: 
```
DATASET_PATH = os.path.join(os.environ.get('SM_CHANNEL_TRAINING'), 'dataset_for_training_risk_level_5.h5')
```
That variable will actually have the value `'/opt/ml/input/train/dataset_for_training_risk_level_5.h5'`.



### Checking container
Again, you can verify your container interactively with 
```
docker run -it --rm 931683541123.dkr.ecr.us-west-2.amazonaws.com/training-example11
```
You can find the container URI on ECR in AWS's website.

If you run into errors with the container, the easiest is to create a new container by change the name in `adding_code/create_container.sh` line 4. 

Don't forget to delete unused containers on ECR in AWS's website as you will incur costs for those even if you don't use them.

Cost saving tip: Do store large files in your container. Your container should only contain your code. Large file such as dataset will be store in S3. Storage costs are much higher on ECR than on S3.

## 5. Run in Local Mode

First, I suggest debugging your scripts with `pdb` without using SageMaker. Then make sure that your scripts run in local mode. You can debug your scripts locally using SageMake and print statements by repeating step 4 and 5. When that works, running on the cloud (step 6) is usually seamless. You do not want to debug directly on the cloud because provisioning ressources takes time. Repeating step 4 with small code modifications is usually very fast as unchanged data is automatically cached. 

### Execution Role

Find your sagemaker role string.

You can check on the web platform (SageMaker page, right, under Domain panel).

Or, simpler:

`aws iam list-roles|grep SageMaker-Execution`

### Configuration

Open the script `run_local/run_sagemaker.py.`

You need to change four strings:

1. `role` (line 13) as explained above
2. DATA IN: the string as input to `estimator.fit` (line 27). If you try to pass an S3 uri, it will crash. So pass a local path with the keyword `'file://'` like in the example. This path cannot be a relative path.
3. DATA OUT: the string argument `output_path` of the estimator during declaration. This path cannot be a relative path.
4. The docker image string/uri as argument `image_uri` of the estimator during declaration. You can find that in ECR in the AWS website.

## 6. Run on the Cloud

### Push data to s3

First create a bucket using the web platform, then

`aws s3 sync my_folder s3://bucker_name/my_folder`

example:

`aws s3 sync data/2 s3://sagemaker-training-example/data/2`

### Configuration

Open the script `run_cloud/run_sagemaker.py`.

You need to change six strings:

1. `role` (line 13) as explained above
2. DATA IN: the string as input to `estimator.fit` (line 27), as an s3 uri
3. DATA OUT: the string argument `output_path` of the estimator during declaration, as an S3 uri
4. The docker image string/uri as argument `image_uri` of the estimator during declaration
5. Choose as string for `base_job_name` as argument the estimator during declaration. This string cannot have underscore char `_` or SM will crash. If you do not specify `base_job_name`, a job name will automatically be created using your docker image name
6. Choose `instance_type` as argument the estimator during declaration

You can find all information related to your training job (GPU usage, metrics, and so) on the AWS web plateform. Metrics will be tracked with CloudWatch.

To track your own metric, you need to parse the logs using a regex. That's what I do in `run_cloud/run_sagemaker.py`:
```
metric_definitions=[
        {'Name': 'val_loss', 'Regex': 'val_loss: ([0-9\\.]+)'}
    ]
```
Under `Name`, you can set any string. It does have to correspond to your logs. This is how your metric will appear on the AWS web plateform.
Under `Regex`, you need to define the regex as it appears in your logs. AWS accepts specific regex so it takes so trial and error to figure out the corect expression. For floats, the regex in the example above should work.

## Retrieve Data from S3

Once your algorithm finished training, you can retrieve models and logs from S3. The easiest is to use the CLI for that:

`aws s3 cp s3://bucket_name/my_folder my_folder --recursive`

Example:

`aws s3 cp s3://sagemaker-training-example/output-sagemaker-ready/ test_sagemaker_cloud_outputs --recursive`

You can also retrieve data from S3 using boto3 for python. But usually, you will need more line of code than using the CLI.

## 7. Hyperparameter Tuning

With SageMaker, you can perform automatic hyperparameters optimization with bayesian search (default). You can also chose other types of optimization (like grid search), but bayesian search has been shown to be more efficient.

You need to modify two scripts. Your `train.py` script, and the `run_sagemaker.py` that we use to launch the jobs.

The trainining script `adding_code/container/code/cifar/network/segmentation_basis/train.py` should accept arguments using `argparse`:

```
import argparse

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--aug_shift', type=float, default=0.05)
parser.add_argument('--aug_rot', type=float, default=0.1)
parser_args, _ = parser.parse_known_args()
```

In launch script `hyperparameter_tuning/run_sagemaker.py`, you should define the hyperparameters types and ranges:
```
hyperparameter_ranges = {
    "lr": ContinuousParameter(1e-7, 1e-3),
    "aug_shift": ContinuousParameter(0., 0.3),
    "aug_rot": ContinuousParameter(0., 0.4),
}
```
You should define your target metric using regex, similar to what was explained in the previous section:
```
objective_metric_name = "val_loss"
objective_type = "Minimize"
metric_definitions = [{'Name': 'val_loss', 'Regex': 'val_loss: ([0-9\\.]+)'}]
```
Finally, you should create another object of the class `HyperparameterTuner`. As input to the constructor, this object needs the `estimator` object as defined in the previous section in `run_sagemaker.py`. It also need the metrics and hyperparameters ranges define above, the max number of jobs for the search and max number of simultaneous jobs.

This results in the following script:
```
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

```

As for the standards training jobs, you can see all information about your hyperparameter tuning job on the AWS web plateform, in the SageMaker section.


## Potential Bugs (not formatted)

This is the last section. More a dump of potential error messages and clues for how to solve them than an organized section.

### RuntimeError: Failed to run: [&#39;docker-compose&#39;, &#39;-f&#39;, &#39;/tmp/tmpxpk0l64a/docker-compose.yaml&#39;, &#39;up&#39;, &#39;--build&#39;, &#39;--abort-on-container-exit&#39;], Process exited with code:

You need to pull the full uri in the image\_uri argument of the sagemaker estimator e.g.

763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15-cpu-py3

And not only

Tensorflow-training:1.15-cpu-py3

### Automatic requirements install

vmzfevtjfe-algo-1-3twxb | 2022-06-21 03:29:20,928 sagemaker-training-toolkit INFO Installing dependencies from requirements.txt:

vmzfevtjfe-algo-1-3twxb | /usr/local/bin/python3.9 -m pip install -r requirements.txt

vmzfevtjfe-algo-1-3twxb | Processing /tmp/build/80754af9/absl-py\_1623861369333/work

vmzfevtjfe-algo-1-3twxb | ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: &#39;/tmp/build/80754af9/absl-py\_1623861369333/work&#39;

**If you have**  **requirements.txt**  **in the code folder, sagemaker will automatically try to install it at launch**

### Entrypoint in dockerfile and local mode

If you define an entrypoint in the dockerfile you cannot have access to SM env variables or hyperparameters in local mode. (it is a bug)

[https://github.com/aws/sagemaker-python-sdk/issues/2930](https://github.com/aws/sagemaker-python-sdk/issues/2930)

### SM\_TRAINING\_ENV is a str (not a dict)

To read as dict, converting with ast does not seem to work. Not sure why.

To access the training folder, just use the default os.environ.get(&#39;SM\_CHANNEL\_TRAINING&#39;)

### at &#39;trainingJobName&#39; failed to satisfy constraint: Member must satisfy regular expression pattern: ^[a-zA-Z0-9](-\*[a-zA-Z0-9]){0,62}

The training job name (and consequently container tag) cannot contain underscores.

If you don&#39;t want to change your container name, you can specify a training job name to the SM estimator (_**base\_job\_name).**_
