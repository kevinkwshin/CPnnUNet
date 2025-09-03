#!/bin/bash

###############################################################################
# Configuration
###############################################################################
gpu_id=0 # GPU_ID
input_folder="/workspace/data/project/mbh-seg/MBH_Val_2025_voxel-label" # input images path
output_folder="./MBHSEG2025_outputs" # output folder path
################################################################################

################################################################################
# Do not change
################################################################################
input_folder_="./MBHSEG2025_inputs"

annotator_id=3 
dataset_id=802 
fold="all" 
configuration="3d_fullres" # don't change this setting
model_folder="./nnunetv2/weight/Dataset802_MBHMultiAnnot/ProbabilisticUNetTrainer__nnUNetPlans__${configuration}"
checkpoint="checkpoint_final.pth"

export nnUNet_raw=None
export nnUNet_preprocessed=None
export nnUNet_results=./nnUNet_results

# Setup
pip install .

# Preprocessing
python preprocess.py --src_root ${input_folder} --dst_root ${input_folder_}

# Inference
SAMPLING_SCALE=2.0 \
TORCH_COMPILE=0 \
OPENBLAS_NUM_THREADS=8 \
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=${gpu_id} \
PYTHONPATH=/workspace/data/project/PnnUNet \
python nnunetv2/inference/predict_from_raw_data.py \
    -i ${input_folder_} \
    -o ${output_folder} \
    -m ${model_folder} \
    -f ${fold} \
    -d cuda \
    -c ${configuration} \
    -annotator_id ${annotator_id}

################################################################################
