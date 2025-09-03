import os
import shutil
import argparse

def main(src_root, dst_root):
    # 목적지 폴더 없으면 생성
    os.makedirs(dst_root, exist_ok=True)

    # 하위폴더 순회
    for folder_name in os.listdir(src_root):
        folder_path = os.path.join(src_root, folder_name)
        if os.path.isdir(folder_path):
            src_file = os.path.join(folder_path, "image.nii.gz")
            if os.path.exists(src_file):
                dst_file = os.path.join(dst_root, f"{folder_name}_0000.nii.gz")
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} -> {dst_file}")
            else:
                print(f"⚠️ No image.nii.gz in {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy image.nii.gz files and rename with folder name.")
    parser.add_argument("--src_root", type=str, required=True, help="Source root directory (e.g., MBH_val_label_2025)")
    parser.add_argument("--dst_root", type=str, required=True, help="Destination directory (e.g., output_dir)")
    
    args = parser.parse_args()
    main(args.src_root, args.dst_root)