#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=two_opponents_at_once_40k

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=23:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=torsten.foehr@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

# In this case, none.

####
#c) Execute your  code in a specific singularity container
####

##################################################################################################
# TODO: update the following paths with your own!
singularity exec --nv ~/Singularity_Torsten python3 ~/RL/RL_Agent/DDPG/TD3.py --mode "train" --agent "two_opponents_at_once_40k" --config "~/RL/RL_Agent/DDPG/jobs/two_opponents_at_once_40k.json"
##################################################################################################
