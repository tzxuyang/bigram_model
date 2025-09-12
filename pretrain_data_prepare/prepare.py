# https://huggingface.co/datasets/Skylion007/openwebtext/tree/main
# concate the txt file from above path to a large txt file
import tarfile
import os
import lzma

# Define the path to your tar file
tar_file_path = "./urlsf_subset01.tar"  # Replace with your actual file path

# Define the directory where you want to extract the contents
destination_directory = "./extract/"  # Replace with your desired destination

def get_all_files_in_directory(directory_path):
    """
    Retrieves a list of all file paths within a given directory,
    including files in subdirectories.

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        list: A list of absolute paths to all files found.
    """
    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def decompress_xz_file(filepath):
    """Decompresses a standalone .xz file."""
    input_filepath = filepath
    output_filepath = filepath.replace("xz","txt")
    with lzma.open(input_filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(input_filepath)
    

def extract_files(tar_file_path, destination_directory):
    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    try:
        # Open the tar file in read mode ('r')
        # The 'r' mode handles various compression types (gz, bz2, xz) automatically
        with tarfile.open(tar_file_path, 'r') as tar:
            # Extract all members to the specified destination directory
            tar.extractall(path=destination_directory)
        print(f"Archive '{tar_file_path}' extracted successfully to '{destination_directory}'.")
    except tarfile.ReadError:
        print(f"Error: Could not open or read the tar file '{tar_file_path}'. It might be corrupted or not a valid tar archive.")
    except FileNotFoundError:
        print(f"Error: The tar file '{tar_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def concate_txt(file_list, output_filename):
    # List of filenames to concatenate

    with open(output_filename, 'w') as outfile:
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            # Optional: Add a newline between files for better readability
            outfile.write('\n') 

if __name__=="__main__":
    # os.remove("./extract/openwebtext")

    target_files = [f"./urlsf_subset0{i+1}.tar" for i in range(2)]
    print(target_files)
    for target_file in target_files:
        extract_files(target_file, "./extract")
    
    file_path = "./extract/openwebtext"
    file_list = get_all_files_in_directory(file_path)
    print(file_list)

    for file in file_list:
        decompress_xz_file(file)

    file_list = get_all_files_in_directory(file_path)
    print(file_list)
    output_file = "../input_pre.txt"
    concate_txt(file_list, output_file)


