import os
from tqdm import tqdm

def copy_and_rename_files_in_subdir(new_video_dir):
    for root, dirs, files in tqdm(os.walk(new_video_dir)):
        for file in files:
            # Construct the full file path
            old_file_path = os.path.join(root, file)
            
            new_file_name = file.replace("_raw_data", "")
            new_file_path = os.path.join(root, new_file_name)
            
            os.rename(old_file_path, new_file_path)
            print(f"Copied and renamed: {old_file_path} -> {new_file_path}")



input_folder = ... # './datasets/street_obstacle_sequences'
new_video_dir = input_folder.replace('raw_data', 'raw_data_tmp')

# We assume that there exists a dir named raw_data_tmp with content copied from raw_data
if not os.path.exists(new_video_dir):
    # Try running the following line,
    # shutil.copytree(video_dir, new_video_dir, dirs_exist_ok=True)
    # or better, copy raw_files from terminal:
    # cp -r raw_data raw_data_tmp
    raise FileNotFoundError(f"The directory {input_folder} does not exist.")

copy_and_rename_files_in_subdir(new_video_dir)