
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.8.0-gpu-py39-cu112-ubuntu20.04-sagemaker

COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt
RUN pip install --upgrade numpy
