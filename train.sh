#!/encs/bin/tcsh

#SBATCH --job-name Swin-PMNet-Train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yjxyang2@outlook.com
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

MVS_TRAINING="/home/mvs_training/dtu/"

# # Train on converted DTU training set
# python train_dtu.py --batch_size 4 --epochs 8 --num_light_idx 7 --input_folder=$MVS_TRAINING --output_folder=$MVS_TRAINING \
# --train_list=lists/dtu/train.txt --test_list=lists/dtu/val.txt "$@"

# Legacy train on DTU's training set
python train_dtu.py --batch_size 4 --epochs 8 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt \
--vallist lists/dtu/val.txt --logdir ./checkpoints "$@"
