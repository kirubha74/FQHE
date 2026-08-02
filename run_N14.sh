#!/bin/bash
#SBATCH --job-name=laughlin_N14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=sf_N14_%j.out
#SBATCH --error=sf_N14_%j.err

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=close
export OMP_PLACES=cores

g++ -std=c++23 -O3 -march=native -fopenmp \
    -o laughlin_N14 Laughlin_structure_factor_N14.cpp || exit 1

echo "start: $(date)"
numactl --interleave=all ./laughlin_N14
echo "end:   $(date)"
