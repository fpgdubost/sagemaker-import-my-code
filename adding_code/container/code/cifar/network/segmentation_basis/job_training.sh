#!/bin/bash
#SBATCH --job-name=patting_detection.job
#SBATCH --output=logs/%j_seg_real_data.out
#SBATCH --error=logs/%j_seg_real_data.err
#SBATCH --time=168:00:0
#SBATCH --qos=normal
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# first activate virtual environment 
source /share/pi/cleemess/fdubost/eeg_video/scripts/virutalEnvs/cnn3/bin/activate

# load in the proper cuda environment 
module load cuda 

# alter your variables to get access to cuda and libraries 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/sw/open/cuda/10.1/lib64 

# call the training function
experimentId=$1
cd /share/pi/cleemess/fdubost/eeg_video/scripts/patting_network/segmentation_basis
python3 train.py $experimentId
