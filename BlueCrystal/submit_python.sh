#!/bin/bash
#SBATCH --job-name=viscek_wall
#SBATCH --partition=teach_cpu
#SBATCH --account=chem033284
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=0:02:00
#SBATCH --mem-per-cpu=100M

## Direct output to the following files.
## (The %j is replaced by the job id.)
#SBATCH -e viscek_err_%j.txt
#SBATCH -o viscek_out%j.txt


# Load Python module
module add languages/python/3.12.3

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}" 
printf "\n\n"

# Run the Python script
python "Vicsek_loops_wall optimising.py"

# Output the end time
printf "\n\n"
echo "Ended on: $(date)"

