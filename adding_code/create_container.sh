#!/bin/bash

# The name of our algorithm
algorithm_name=training-example11

cd container

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password | docker login -u AWS --password-stdin "https://${account}.dkr.ecr.${region}.amazonaws.com"
# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}

# clear all untagged images in repo
IMAGES_TO_DELETE=$( aws ecr list-images --region ${region} --repository-name "${algorithm_name}" --filter "tagStatus=UNTAGGED" --query 'imageIds[*]' --output json )
aws ecr batch-delete-image --region ${region} --repository-name "${algorithm_name}" --image-ids "$IMAGES_TO_DELETE" || true