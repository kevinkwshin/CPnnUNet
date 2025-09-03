import argparse
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results # nnUNet_results 추가
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

# Directly import our custom trainer
from nnunetv2.training.nnUNetTrainer.variants.probabilistic_trainer.ProbabilisticUNetTrainer import ProbabilisticUNetTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str, help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str, help="Configuration that should be trained")
    parser.add_argument('fold', type=str, help='Fold of the 5-fold cross-validation. Can be an integer or all')
    parser.add_argument('-device', type=str, default='cuda', help="Use 'cuda' or 'cpu'")
    parser.add_argument('-c', '--continue_training', action='store_true', help="Continue training from the latest checkpoint.")
    args = parser.parse_args()

    # --- Convert fold to int if it's not 'all' --- #
    if args.fold != 'all':
        fold = int(args.fold)
    else:
        fold = 'all'

    # --- 1. Setup paths and load plans/dataset JSON --- #
    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, dataset_name)
    plans_file = join(preprocessed_dataset_folder_base, 'nnUNetPlans.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))

    # --- 2. Instantiate the trainer directly --- #
    print(f"\nInstantiating trainer: ProbabilisticUNetTrainer")
    trainer = ProbabilisticUNetTrainer(
        plans=plans,
        configuration=args.configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=torch.device(args.device)
    )

    # --- 3. Load checkpoint if continue_training is True --- #
    if args.continue_training:
        # Construct the full path to the latest checkpoint file
        checkpoint_path = join(trainer.output_folder, 'checkpoint_latest.pth')
        print(f"Continuing training from latest checkpoint: {checkpoint_path}...")
        trainer.load_checkpoint(checkpoint_path)

    # --- 4. Run training --- #
    print(f"Starting training for {dataset_name} fold {args.fold}...")
    trainer.run_training()

if __name__ == '__main__':
    main()
