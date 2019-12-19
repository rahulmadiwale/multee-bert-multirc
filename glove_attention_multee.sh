#!/bin/bash
#SBATCH -p gpu                   # Asking to assign "gpu" partition(queue)
#SBATCH --gres=gpu         # Asking to assign one GPU. You can ask two or more GPUs, but unless your project is coded to run parallel, it will only waste resources.
         # Asking time. "gpu" currently let you run an experiment upto 36 hours.

# Activating conda environment to run tensorflow (python 2.7)
source /home/rmadiwale/advanced_project/multee/bin/activate

# Change the working directory to the project directory
# cd /home/rmadiwale/nlp-question-answering/

# Run the experiment.
allennlp train configs/multee_with_glove_attention.jsonnet -s glove --include-package lib