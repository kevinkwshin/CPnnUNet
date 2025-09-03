import sys
import os

# Add the project root to the Python path
project_root = '/workspace/data/project/PnnUNet'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from nnunetv2.training.nnUNetTrainer.variants.probabilistic_trainer.ProbabilisticUNetTrainer import ProbabilisticUNetTrainer
    print("Successfully imported ProbabilisticUNetTrainer!")
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"sys.path: {sys.path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
