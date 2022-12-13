#!/bin/bash
#SBATCH --job-name=patting_detection.job
#SBATCH --output=logs/%j_seg_real_data.out
#SBATCH --error=logs/%j_seg_real_data.err
#SBATCH --time=168:00:0
#SBATCH --qos=normal
#SBATCH --mem=MaxMemPerNode
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


# arg
experimentId=$1
path_cnn=$2
path_samples=$3

# intermediary variable
root_savepath='/share/pi/cleemess/fdubost/eeg_video/experiments'

# bash: declare -i segment_id=0
# fish: set segment_id 0
for line in `cat $path_samples/*.csv`; do
    segment_id="${line%%,*}"
    echo $segment_id

    # extract single segment
    python save_samples_as_individual_files.py $experimentId $path_samples $segment_id

    # input parameters
    input_file=${root_savepath}/${experimentId}/single_sample.npy.gz
    output_file=${root_savepath}/${experimentId}/prediction_${segment_id}.csv
    attention_map_file=${root_savepath}/${experimentId}/attention_map_${segment_id}.pdf
    CNN_architecture=${path_cnn}/model.json
    CNN_weights=${path_cnn}/best_weights.hdf5

    python test.py $input_file $output_file $attention_map_file $CNN_architecture $CNN_weights

    # indent loop variable
    # bash: segment_id+=1
    # set segment_id (math $segment_id + 1)
done

