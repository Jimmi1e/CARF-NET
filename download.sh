#!/encs/bin/tcsh

#SBATCH --job-name=downloadDataset       ## Give the job a name
#SBATCH --mail-type=ALL        ## Receive all email type notifications
#SBATCH --mail-user=yjxyang2@outlook.com
#SBATCH -o output-%A.log
#SBATCH --chdir=./             ## Use currect directory as working directory

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      ## Request 1 cpus
#SBATCH --mem=1G               ## Assign 1G memory per node

echo "Downloading DTU dataset..."
echo "=============================" 
sleep 30
date 
wget https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35/download
date
echo "Finished"