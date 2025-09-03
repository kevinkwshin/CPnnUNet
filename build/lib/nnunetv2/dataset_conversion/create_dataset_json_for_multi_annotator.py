import argparse
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def main():
    parser = argparse.ArgumentParser(description="Create dataset.json for a dataset with multiple annotators per image.")
    parser.add_argument("-d", "--dataset_name", type=str, help="Name of the dataset. Used to create the output folder.")
    parser.add_argument("-i", "--imagesTr_dir", type=str, help="Path to the folder containing the training images.")
    parser.add_argument("-l", "--labelsTr_dir", type=str, help="Path to the folder containing the training labels.")
    parser.add_argument("-c", "--channel_names", nargs="+", help="Names of the channels/modalities.", required=True)
    parser.add_argument("-s", "--segmentation_labels", nargs="+", help="Names of the segmentation labels.", required=True)
    args = parser.parse_args()

    # Create the output directory in the nnUNet_raw folder
    output_dir = Path(nnUNet_raw) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    images_tr_dir = Path(args.imagesTr_dir)
    labels_tr_dir = Path(args.labelsTr_dir)

    # --- Create dataset.json ---
    training_cases = []
    image_files = sorted(subfiles(images_tr_dir, suffix=".nii.gz", join=False))

    for img_file in image_files:
        # Extract the case identifier (e.g., "prostate_01" from "prostate_01_0000.nii.gz")
        case_id = "_".join(img_file.split("_")[:-1])
        
        # Find all corresponding label files for this case
        # This assumes labels are named like "case_id_annotator1.nii.gz", "case_id_annotator2.nii.gz"
        annotator_label_files = sorted(subfiles(labels_tr_dir, prefix=case_id, suffix=".nii.gz", join=False))
        
        if not annotator_label_files:
            print(f"Warning: No labels found for image {img_file}. Skipping.")
            continue

        # Create relative paths for the json file
        image_path_relative = f"./{images_tr_dir.name}/{img_file}"
        label_paths_relative = [f"./{labels_tr_dir.name}/{f}" for f in annotator_label_files]

        training_cases.append({
            "image": image_path_relative,
            "label": label_paths_relative if len(label_paths_relative) > 1 else label_paths_relative[0]
        })

    # Create channel_names and labels dictionaries for generate_dataset_json
    channel_dict = {str(i): name for i, name in enumerate(args.channel_names)}
    
    # Background is 0, then labels start from 1
    labels_dict = {"background": 0}
    for i, label_name in enumerate(args.segmentation_labels):
        labels_dict[label_name] = i + 1

    # Use the imported generate_dataset_json function
    generate_dataset_json(
        str(output_dir),
        channel_dict,
        labels_dict,
        len(training_cases),
        ".nii.gz",
        dataset_name=args.dataset_name,
        description="Dataset with multiple annotators per image.",
        training=training_cases  # Pass the training cases to be included in the json
    )

    # --- The script assumes you will manually create the symlinks or copy the files ---
    print(f"Successfully created dataset.json at {output_dir / 'dataset.json'}")
    print("\nIMPORTANT:")
    print("This script ONLY creates the dataset.json file.")
    print(f"You must manually create symbolic links or copy your images and labels to the following directories:")
    print(f"Images: {output_dir / images_tr_dir.name}")
    print(f"Labels: {output_dir / labels_tr_dir.name}")


if __name__ == "__main__":
    main()
