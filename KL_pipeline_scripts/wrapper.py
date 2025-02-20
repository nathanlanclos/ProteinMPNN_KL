#!/usr/bin/env python3
import subprocess
import numpy as np
import os

def run_scripts(scripts_with_args):
    """
    Launch multiple Python scripts concurrently.
    
    Each element in scripts_with_args can be either:
      - A tuple: (script, args_list), where args_list is a list of command-line arguments.
      - A string: the script filename (if no arguments are needed).
    """
    processes = []
    for item in scripts_with_args:
        if isinstance(item, tuple):
            script, args = item
        else:
            script = item
            args = []
        cmd = ["python3", script] + args
        print("Starting command:", " ".join(cmd))
        proc = subprocess.Popen(cmd)
        processes.append(proc)
    
    for proc in processes:
        proc.wait()
    print("All scripts have completed.")

def combine_specific_files(files, folder, output_filename="combined.npy"):
    """
    Load the specified files (a mix of .npy and .csv) from the provided folder,
    combine them (after ensuring each is at least 1D) using np.concatenate along axis 0,
    and save the result as an NPY file.
    
    Also output a human-readable CSV file.
    
    Parameters:
      - files: list of filenames (e.g. ["plddt_positions.npy", "protein_masked_position_30.csv"])
      - folder: the directory where the files are located.
      - output_filename: name for the combined NPY output file.
    """
    arrays = []
    for file in files:
        file_path = os.path.join(folder, file)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue
        print(f"Loading {file_path}...")
        try:
            if file.lower().endswith('.csv'):
                # Load CSV data (assume a comma-separated list)
                data = np.genfromtxt(file_path, delimiter=',')
            elif file.lower().endswith('.npy'):
                data = np.load(file_path, allow_pickle=True)
            else:
                print(f"Unknown file extension for {file_path}. Skipping.")
                continue
            # Ensure the loaded data is at least 1D.
            data = np.atleast_1d(data)
            arrays.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not arrays:
        print("No valid files were loaded from", folder)
        return
    
    try:
        combined = np.concatenate(arrays, axis=0)
    except Exception as e:
        print(f"Error concatenating arrays: {e}")
        combined = np.array(arrays, dtype=object)
    
    # Save the combined data as an NPY file.
    output_path = os.path.join(folder, output_filename)
    np.save(output_path, combined)
    print(f"Combined NPY file saved as {output_path}")
    
    # Also save a human-readable CSV version.
    csv_filename = os.path.splitext(output_filename)[0] + ".csv"
    output_csv_path = os.path.join(folder, csv_filename)
    try:
        np.savetxt(output_csv_path, combined, delimiter=",", fmt="%s")
        print(f"Combined CSV file saved as {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    # Define the common output folder where both scripts write their output.
    output_folder = "/Users/marcohuang/Desktop/Keasling_Lab/ProteinMPNN_KL/KL_pipeline_scripts/testresults"
    
    # Define the scripts to run along with their command-line arguments.
    scripts_to_run = [
        (
            "npy_writer.py", 
            [
                "--fasta", "/Users/marcohuang/Desktop/Keasling_Lab/MSAReader/Inputs/combined_test_2.fasta",
                "--output", output_folder,
                "--protein_of_interest", "/Users/marcohuang/Desktop/Keasling_Lab/MSAReader/Inputs/test_WT.fasta",
                "--threshold", "30"
            ]
        ),
        (
            "extract_plddt_npy.py", 
            [
                "--input_path", "/Users/marcohuang/Desktop/Keasling_Lab/ProteinMPNN_KL/KL_pipeline_scripts/test", 
                "--output_dir", output_folder,
                "--plddt_range", "0-80",
                "--log_dir", "logs"
            ]
        )
    ]
    
    # Run the specified scripts.
    run_scripts(scripts_to_run)
    
    # Specify the files to combine:
    # One NPY file and one CSV file (the CSV is assumed to be a comma-separated list of numbers)
    files_to_combine = ["plddt_positions.npy", "protein_masked_position_30.csv"]
    
    # Combine the files into one NPY file and also output a human-readable CSV file.
    combine_specific_files(files_to_combine, output_folder, output_filename="combined.npy")
