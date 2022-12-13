#!/bin/bash

path_cnn=/share/pi/cleemess/fdubost/eeg_video/experiments/124
samples_paths=($(seq 231 243))
experimentsIds=($(seq 244 256))

for (( i=0; i<${#samples_paths[@]}; i++ )); do
  samples_path=${samples_paths[$i]}
  experimentsId=${experimentsIds[$i]}
  sbatch predict_full_video_direct.sh $experimentsId $path_cnn /share/pi/cleemess/fdubost/eeg_video/experiments/${samples_path}
done