# MBH Segmentation 2025

This repository contains the inference script for MBH (Multi-annotator Brain segmentation) segmentation using Conditional Probabilistic nnU-Net framework.

## Prerequisites

- CUDA-compatible GPU
- Python environment with required dependencies
- nnU-Net framework
- ProbabilisticUNet implementation

## Usage

### Step 1: Set Up Environment

Install the required dependencies:

```bash
pip install .
```

This will install all necessary packages for the MBH segmentation pipeline.


### Step 2: Configure Input/Output Paths

Open `run_inference.sh` with an editor and edit the configuration section at the top of the script:

1. Set `gpu_id` to your desired GPU (default: 0)
2. Set `input_folder` to your input data directory
3. Set `output_folder` to your desired output directory


### Step 3: Run the Script

Make the script executable and run:

```bash
chmod +x run_inference.sh
./run_inference.sh
```


### Step 4: Evaluate the Results

Evaluate your results based on the files generated in the `output_folder` directory. The segmentation results will be saved in the specified `output_folder` directory.


## Important Notes

- Only modify the parameters in the "Configuration" section
- Do not change parameters marked as "Do not change"
- Ensure the model checkpoint file exists in the specified location
- The script expects specific directory structure for nnU-Net results