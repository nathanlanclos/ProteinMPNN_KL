#!/usr/bin/env python

import os
import argparse
import glob
import logging
import numpy as np

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def chains_are_identical(structure):
    """
    Check if all protein chains in the structure have the same
    amino acid sequence (ignoring hetero residues).

    Returns:
        bool: True if all chains have the same sequence, False otherwise.
    """
    sequences = []
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                # Skip hetero residues or non-standard
                if residue.id[0].strip():
                    continue  
                # Attempt to parse standard residue name -> 1-letter code
                # If 'CA' is not present, skip
                if 'CA' not in residue:
                    continue
                aa_3 = residue.get_resname().strip()
                try:
                    # Convert 3-letter code to 1-letter code
                    aa_1 = seq1(aa_3)
                except:
                    # For unusual residues that Biopython doesn't recognize
                    aa_1 = 'X'
                seq.append(aa_1)
            if seq:
                sequences.append("".join(seq))
        # For simplicity, we only check the first model if you have multiple models.
        break

    if not sequences:
        # No standard chains found
        return True  # trivially "same" if none found
    # Compare all sequences to the first one
    return all(s == sequences[0] for s in sequences[1:])


def parse_first_chain_plddt_positions(pdb_file, plddt_start, plddt_end):
    """
    Parse a single PDB file using Biopython, but only look at the *first chain*
    in the first model. Extract pLDDT (B-factor) from each residue's C-alpha
    atom, and return the list of residue IDs whose pLDDT is in [plddt_start, plddt_end].
    
    Returns:
        positions_in_range (list of int): PDB residue numbers matching the range.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)

    # Take only the first model and first chain
    model = next(structure.get_models())
    chain = next(model.get_chains())

    positions_in_range = []
    for residue in chain:
        # Skip hetero residues
        if residue.id[0].strip():
            continue
        # Make sure we have a CA atom
        if 'CA' not in residue:
            continue

        ca_atom = residue['CA']
        plddt_value = ca_atom.get_bfactor()

        if plddt_start <= plddt_value <= plddt_end:
            # residue.id is typically something like (' ', 50, ' ')
            residue_number = residue.id[1]
            positions_in_range.append(residue_number)

    return positions_in_range


def main(args):
    """
    Main function that:
      - Determines if input_path is a file or directory
      - Parses only the *first chain* for each PDB
      - Checks if all chains are identical in sequence
      - Filters residues by the specified pLDDT range
      - Saves results (.npy) with "positions" and "all_chains_same" per file
      - Optionally logs progress
    """
    # 1. Set up logging
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, "script.log")
        logging.basicConfig(filename=log_file, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("Starting pLDDT extraction script (monomer mode).")

    # 2. Identify input files
    if os.path.isfile(args.input_path):
        # Single PDB
        pdb_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        # Directory of PDBs
        pdb_files = glob.glob(os.path.join(args.input_path, "*.pdb"))
    else:
        raise ValueError(f"ERROR: {args.input_path} is neither a file nor a directory.")

    if not pdb_files:
        logging.warning("No PDB files found in the given input path.")
        return

    # 3. Parse pLDDT range (e.g., "0-80" => 0 and 80)
    try:
        plddt_start_str, plddt_end_str = args.plddt_range.split("-")
        plddt_start, plddt_end = float(plddt_start_str), float(plddt_end_str)
    except Exception as e:
        raise ValueError(
            f"Invalid --plddt_range format: {args.plddt_range}. "
            "Expected something like '0-80' or '70-100'."
        ) from e

    # 4. Process each PDB file
    results_dict = {}
    parser = PDBParser(QUIET=True)

    for pdb_file in pdb_files:
        logging.info(f"Parsing {pdb_file} (first chain only)...")
        
        # Parse only the first chain's residues in range
        positions = parse_first_chain_plddt_positions(
            pdb_file, plddt_start, plddt_end
        )

        # Check if all chains are the same in this PDB
        structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
        all_same = chains_are_identical(structure)

        # Store results
        results_dict[os.path.basename(pdb_file)] = {
            "positions": positions,
            "all_chains_same": all_same
        }

        logging.info(f"Found {len(positions)} residues in pLDDT range {plddt_start}-{plddt_end}. "
                     f"All chains same? {all_same}")

    # 5. Save results as a .npy file in the specified output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "plddt_positions.npy")
    np.save(output_path, results_dict)

    logging.info(f"Saved results to {output_path}")
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract residue positions (first chain only) from PDBs where pLDDT (B-factor) is in a given range, and check if all chains match."
    )
    
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to a single .pdb file or a directory containing multiple .pdb files."
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Folder to save the .npy results file."
    )
    parser.add_argument(
        "--plddt_range",
        default="70-100",
        help="Range of pLDDT values to filter on, e.g. '0-80', '60-70', '70-100'."
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="Optional folder to store 'script.log'. If empty, logs go to stdout."
    )

    args = parser.parse_args()
    main(args)
