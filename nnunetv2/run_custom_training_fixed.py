import argparse
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

# Directly import our custom trainer
from nnunetv2.training.nnUNetTrainer.variants.probabilistic_trainer.ProbabilisticUNetTrainer_fixed import ProbabilisticUNetTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str, help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str, help="Configuration that should be trained")
    parser.add_argument('fold', type=int, help='Fold of the 5-fold cross-validation.')
    parser.add_argument('-device', type=str, default='cuda', help="Use 'cuda' or 'cpu'")
    args = parser.parse_args()

    # --- 1. Setup paths and load plans/dataset JSON ---
    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, dataset_name)
    plans_file = join(preprocessed_dataset_folder_base, 'nnUNetPlans.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))

    # --- 2. Instantiate the trainer directly ---
    print(f"\nInstantiating trainer: ProbabilisticUNetTrainer")
    trainer = ProbabilisticUNetTrainer(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        device=torch.device(args.device)
    )

    # --- 3. Run training ---
    print(f"Starting training for {dataset_name} fold {args.fold}...")
    trainer.run_training()

if __name__ == '__main__':
    main()
