#!/bin/bash
#SBATCH -p gpu                   # Asking to assign "gpu" partition(queue)
#SBATCH --gres=gpu         # Asking to assign one GPU. You can ask two or more GPUs, but unless your project is coded to run parallel, it will only waste resources.
         # Asking time. "gpu" currently let you run an experiment upto 36 hours.

# Activating conda environment to run tensorflow (python 2.7)
source /home/rmadiwale/venv/nlp-qa/bin/activate

# Change the working directory to the project directory
# cd /home/rmadiwale/nlp-question-answering/

# Run the experiment.
allennlp train configs/bert_entailment_without_encoder.jsonnet -s bert_entailment --include-package lib