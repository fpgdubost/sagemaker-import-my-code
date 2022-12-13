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
path_list_samples=$3

# intermediary variable
root_savepath='/share/pi/cleemess/fdubost/eeg_video/experiments'
path_list_samples_ids=${path_list_samples}/list_sample_ids.csv


for segment_id in `cat $path_list_samples_ids`; do
    # input parameters
    input_file=${path_list_samples}/${segment_id}.npy.gz
    output_file=${root_savepath}/${experimentId}/prediction_${segment_id}.csv
    attention_map_file=${root_savepath}/${experimentId}/attention_map_${segment_id}.pdf
    CNN_architecture=${path_cnn}/model.json
    CNN_weights=${path_cnn}/best_weights.hdf5

    python test.py $input_file $output_file $attention_map_file $CNN_architecture $CNN_weights
done

