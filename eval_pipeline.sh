#!/encs/bin/tcsh

#SBATCH --job-name PatchmatchNet-eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuhangchen0425@gmail.com
#SBATCH --chdir=./
#SBATCH -o output-%A.log

# Request Resources
#SBATCH --mem=60G
#SBATCH -n 32
#SBATCH --gpus=1
#SBATCH -p pt

# Load required modules
module load anaconda3/2023.03/default
module load cuda/9.2/default

# Define environment name and path 
set ENV_NAME = "project_env"
set ENV_DIR = "/speed-scratch/$USER/envs"
set ENV_PATH = "$ENV_DIR/$ENV_NAME"
set TMP_DIR = "/speed-scratch/$USER/envs/tmp"
set PKGS_DIR = "/speed-scratch/$USER/envs/pkgs"

mkdir -p $ENV_DIR
mkdir -p $TMP_DIR
mkdir -p $PKGS_DIR

setenv TMP $TMP_DIR
setenv TMPDIR $TMP_DIR
setenv CONDA_PKGS_DIRS $PKGS_DIR

# Check if the environment exists
conda env list | grep "$ENV_NAME"
if ($status == 0) then
    echo "Environment $ENV_NAME already exists. Activating it..."
    echo "======================================================"
    conda activate "$ENV_PATH"

    if ($status != 0) then
        echo "Error: Failed to activate Conda environment."
        exit 1
	endif
else
	echo "Creating Conda environment $ENV_NAME at $ENV_PATH..."
    echo "===================================================="
	conda create -y -p "$ENV_PATH" python=3.8 -c conda-forge

	echo "Activating environment $ENV_NAME..."
    echo "==================================="
	conda activate "$ENV_PATH"

	if ($status != 0) then
	    echo "Error: Failed to activate Conda environment."
	    exit 1
	endif
	
	echo "Installing required packages..."
    echo "==============================="
	pip install --upgrade pip
    pip install -r requirements.txt
endif

echo "Conda environemnt summary..."
echo "============================"
conda info --envs
conda list

sleep 30

CHECKPOINT_FILE="./checkpoints/params_000007.ckpt"

# test on DTU's evaluation set
DTU_TESTING="/home/dtu/"
echo "Running eval processing..."
echo "================================================"
srun python eval.py --scan_list ./lists/dtu/test.txt --input_folder=$DTU_TESTING --output_folder=$DTU_TESTING \
--checkpoint_path $CHECKPOINT_FILE --num_views 5 --image_max_dim 1600 --geo_mask_thres 3 --photo_thres 0.8 "$@"
# srun python yolo_video.py --input video/v1.avi --output video/001.avi #--gpu_num 1

conda deactivate
exit