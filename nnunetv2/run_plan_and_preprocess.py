import argparse
import os
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join, isdir, load_json, listdir
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprint_dataset, plan_experiments, preprocess_dataset
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

def main():
    parser = argparse.ArgumentParser(description="Run nnU-Net planning and preprocessing directly from a script.")
    parser.add_argument("-d", "--dataset_id", type=int, required=True, help="The ID of the dataset to preprocess.")
    parser.add_argument("-p", "--processes", type=int, default=8, help="Number of processes for fingerprinting and preprocessing.")
    parser.add_argument("-c", "--configurations", nargs="+", default=['2d', '3d_fullres'], help="List of configurations to preprocess.")
    parser.add_argument("--clean", action='store_true', help="Overwrite existing fingerprint and preprocessed files.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.")

    args = parser.parse_args()

    dataset_name = maybe_convert_to_dataset_name(args.dataset_id)
    
    print(f"Running for dataset: {dataset_name} (ID: {args.dataset_id}) with {args.processes} processes.")

    # 1. Fingerprint Extraction
    print("\nStep 1: Extracting fingerprint...")
    try:
        # We use our modified fingerprint extractor that can handle multi-annotator labels
        extract_fingerprint_dataset(
            dataset_id=args.dataset_id,
            fingerprint_extractor_class=DatasetFingerprintExtractor,
            num_processes=args.processes,
            check_dataset_integrity=False,  # Skip the incompatible integrity check
            clean=args.clean,
            verbose=args.verbose
        )
        print("Fingerprint extracted successfully.")
    except Exception as e:
        print(f"\nERROR during fingerprint extraction: {e}")
        print("Please check the traceback above for details.")
        raise e

    # 2. Experiment Planning
    print("\nStep 2: Planning experiments...")
    try:
        planner = ExperimentPlanner(dataset_name, gpu_memory_target_in_gb=8)
        plan_experiments(planner)
        print("Experiment planning successful.")
    except Exception as e:
        print(f"\nERROR during experiment planning: {e}")
        raise e

    # 3. Preprocessing
    print("\nStep 3: Preprocessing...")
    preprocessed_dir = join(nnUNet_preprocessed, dataset_name)
    plans_files = [f for f in listdir(preprocessed_dir) if f.endswith('.json') and f.startswith('nnUNetPlans')]
    
    if not plans_files:
        raise RuntimeError("No plans files found. Experiment planning might have failed.")

    for pf in plans_files:
        plans_name = pf.replace('.json', '')
        plans = load_json(join(preprocessed_dir, pf))
        
        configurations_to_process = {k: v for k, v in plans['configurations'].items() if k in args.configurations}
        
        if not configurations_to_process:
            print(f"Could not find any of the requested configurations ({args.configurations}) in {pf}. Skipping.")
            continue

        print(f"Preprocessing for plans: {plans_name}, configurations: {list(configurations_to_process.keys())}")
        try:
            preprocessor = DefaultPreprocessor()
            for cfg in configurations_to_process.keys():
                print(f"Preprocessing configuration: {cfg}...")
                preprocessor.run(
                    dataset_name_or_id=args.dataset_id,
                    configuration_name=cfg,
                    plans_identifier=plans_name,
                    num_processes=args.processes
                )
        except Exception as e:
            print(f"\nERROR during preprocessing for {plans_name}: {e}")
            raise e
            
    print("\n\nPlanning and preprocessing finished successfully!")

if __name__ == "__main__":
    main()
