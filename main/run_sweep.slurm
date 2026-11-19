#!/bin/bash --login
#SBATCH --job-name=sweep_v4
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=a_XXXX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

IN=${IN:-sweep_v4_in.npz}
OUT=${OUT:-sweep_v4_out.npz}

module load miniforge
eval "$(conda shell.bash hook)"
conda activate fsi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd /scratch/user/$USER/bunya
mkdir -p output
python -u main/main.py "input/$IN" "output/$OUT"