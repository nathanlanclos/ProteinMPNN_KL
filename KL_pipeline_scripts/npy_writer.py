#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
import os
from collections import Counter
import numpy as np

# Mapping from one-letter amino acid codes to three-letter abbreviations.
ONE_TO_THREE = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp',
    'C': 'Cys', 'Q': 'Gln', 'E': 'Glu', 'G': 'Gly',
    'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys',
    'M': 'Met', 'F': 'Phe', 'P': 'Pro', 'S': 'Ser',
    'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
}

def parse_fasta(handle):
    """
    Parse a FASTA file from the given file handle.
    Returns a list of tuples: (header, sequence).
    """
    sequences = []
    header = None
    seq_lines = []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header is not None:
                sequences.append((header, ''.join(seq_lines)))
            header = line
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        sequences.append((header, ''.join(seq_lines)))
    return sequences

def write_temp_fasta(sequences):
    """
    Write the given sequences to a temporary FASTA file.
    Returns the temporary file's name.
    """
    temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".fasta")
    for header, seq in sequences:
        temp.write(f"{header}\n")
        temp.write(f"{seq}\n")
    temp.close()
    return temp.name

def run_clustalo(input_fasta, output_aln):
    """
    Run Clustal Omega to perform a multiple sequence alignment.
    The alignment is written in FASTA format.
    """
    cmd = ["clustalo", "-i", input_fasta, "-o", output_aln, "--force", "--outfmt=fa"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Clustal Omega failed: {e}")

def parse_alignment(aln_filepath):
    """
    Parse an alignment file (in FASTA format) into a list of aligned sequence strings.
    """
    aligned_seqs = []
    with open(aln_filepath, 'r') as f:
        for header, seq in parse_fasta(f):
            aligned_seqs.append(seq)
    if not aligned_seqs:
        raise ValueError("No sequences found in the alignment.")
    # Ensure all sequences are the same length.
    aln_length = len(aligned_seqs[0])
    for s in aligned_seqs:
        if len(s) != aln_length:
            raise ValueError("Aligned sequences are not of equal length.")
    return aligned_seqs

def compute_consensus(aligned_seqs):
    """
    For each column in the alignment, determine the most conserved amino acid (ignoring gaps)
    and its percentage occurrence.
    Returns:
      consensus: the consensus sequence (string)
      conservation: a list of percentages (floats) for each position.
    """
    num_seqs = len(aligned_seqs)
    aln_length = len(aligned_seqs[0])
    consensus = []
    conservation = []
    for pos in range(aln_length):
        column = [seq[pos] for seq in aligned_seqs]
        counter = Counter(column)
        # Ignore gaps when computing consensus.
        non_gap = {res: cnt for res, cnt in counter.items() if res != '-'}
        if non_gap:
            res, cnt = max(non_gap.items(), key=lambda x: x[1])
        else:
            res = '-'
            cnt = counter.get('-', 0)
        percent = (cnt / num_seqs) * 100
        consensus.append(res)
        conservation.append(percent)
    return ''.join(consensus), conservation

def format_consensus_output(consensus, conservation):
    """
    Create a formatted multi-line string showing the consensus sequence and
    per-position conservation percentages.
    """
    output_lines = []
    output_lines.append("Consensus sequence:")
    output_lines.append(consensus)
    output_lines.append("")
    output_lines.append("Position-by-position conservation:")
    for i, (res, perc) in enumerate(zip(consensus, conservation), start=1):
        res_display = ONE_TO_THREE.get(res, res)
        output_lines.append(f"Position {i}: {res_display} : {int(round(perc))}%")
    return "\n".join(output_lines)

def align_two_sequences(seq1, seq2):
    """
    Align two sequences using Clustal Omega.
    Returns a tuple: (aligned_seq1, aligned_seq2).
    """
    # Write the two sequences to a temporary FASTA file.
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".fasta")
    temp_fasta.write(">seq1\n")
    temp_fasta.write(seq1 + "\n")
    temp_fasta.write(">seq2\n")
    temp_fasta.write(seq2 + "\n")
    temp_fasta.close()
    
    # Prepare a temporary file for the alignment.
    temp_aln = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".aln")
    temp_aln.close()
    
    try:
        run_clustalo(temp_fasta.name, temp_aln.name)
        aligned = parse_alignment(temp_aln.name)
        if len(aligned) != 2:
            raise ValueError("Expected an alignment of two sequences.")
        return aligned[0], aligned[1]
    finally:
        os.remove(temp_fasta.name)
        os.remove(temp_aln.name)

def map_conservation(aligned_consensus, aligned_protein, cons_conservation):
    """
    Map the consensus conservation percentages onto the protein sequence.
    For each aligned column:
      - If the consensus has a residue, use the next conservation percentage.
      - If the consensus shows a gap, assign a conservation value of 0.
      - Only record positions where the protein has a residue.
    Returns:
      mapped_residues: list of residues (ungapped) from the protein
      mapped_pcts: list of conservation percentages corresponding to each residue.
    """
    mapped_residues = []
    mapped_pcts = []
    cons_index = 0  # Index into the original (ungapped) consensus conservation list.
    aln_length = len(aligned_consensus)
    for i in range(aln_length):
        cons_char = aligned_consensus[i]
        prot_char = aligned_protein[i]
        if prot_char != '-':  # Only record positions where the protein has a residue.
            if cons_char != '-':
                pct = cons_conservation[cons_index]
            else:
                pct = 0.0
            mapped_residues.append(prot_char)
            mapped_pcts.append(pct)
        if cons_char != '-':
            cons_index += 1
    return mapped_residues, mapped_pcts

def write_masked_outputs_for_threshold(mapped_residues, mapped_pcts, output_folder, threshold):
    """
    For the specified threshold, create three outputs:
      1. A masked protein sequence saved as a NumPy (.npy) file.
         (Filename: protein_masked_<threshold>.npy)
      2. A NumPy file (saved as a text file using np.savetxt) containing the 1-indexed
         positions of residues that are conserved at the specified threshold.
         (Filename: protein_masked_position_<threshold>.npy)
      3. A CSV file containing a comma-separated list of the conserved positions.
         (Filename: protein_masked_position_<threshold>.csv)
    """
    masked = []
    positions = []  # Store the 1-indexed positions for residues meeting the threshold.
    for i, (res, pct) in enumerate(zip(mapped_residues, mapped_pcts)):
        if pct >= threshold:
            masked.append(res)
            positions.append(i + 1)  # 1-indexed position.
        else:
            masked.append('-')
    
    # Save the masked protein sequence as a NumPy (.npy) file.
    masked_array = np.array(masked)
    npy_masked_filename = os.path.join(output_folder, f"protein_masked_{int(threshold)}.npy")
    np.save(npy_masked_filename, masked_array)
    print(f"Saved masked protein sequence (npy) at threshold {threshold}% to {npy_masked_filename}")
    
    # Save the conserved positions as a "numpy" file in human-readable text format.
    # (Instead of np.save, we use np.savetxt with whitespace as the delimiter.)
    positions_array = np.array(positions)
    npy_positions_filename = os.path.join(output_folder, f"protein_masked_position_{int(threshold)}.npy")
    np.savetxt(npy_positions_filename, positions_array, fmt="%d", delimiter=" ")
    print(f"Saved conserved positions (npy, human-readable) at threshold {threshold}% to {npy_positions_filename}")
    
    # Save the conserved positions as a CSV file (comma-separated list).
    csv_positions_filename = os.path.join(output_folder, f"protein_masked_position_{int(threshold)}.csv")
    with open(csv_positions_filename, 'w') as f:
        f.write(",".join(map(str, positions)))
    print(f"Saved conserved positions (csv) at threshold {threshold}% to {csv_positions_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a consensus sequence from a multi-sequence FASTA file using Clustal Omega, "
                    "map its conservation percentages onto a protein of interest, and output (for a specified threshold):\n"
                    "  - consensus.txt\n"
                    "  - protein_masked_(threshold).npy\n"
                    "  - protein_masked_position_(threshold).npy (human-readable, whitespace-separated)\n"
                    "  - protein_masked_position_(threshold).csv"
    )
    parser.add_argument("--fasta", required=True,
                        help="Path to a FASTA file containing multiple sequences for consensus computation.")
    parser.add_argument("--output", required=True,
                        help="Path to the output folder where results will be written.")
    parser.add_argument("--protein_of_interest", required=True,
                        help="Path to the FASTA file for the protein of interest.")
    parser.add_argument("--threshold", type=float, required=True,
                        help="The conservation threshold (e.g., 57 for 57%).")
    args = parser.parse_args()

    # Ensure the output folder exists.
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # --- Consensus Calculation ---
    with open(args.fasta, 'r') as f:
        sequences = parse_fasta(f)
    if not sequences:
        raise ValueError("No sequences found in the provided FASTA file.")

    temp_input = write_temp_fasta(sequences)
    temp_aln_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".aln")
    temp_aln_file.close()
    temp_aln_name = temp_aln_file.name

    try:
        run_clustalo(temp_input, temp_aln_name)
        aligned_seqs = parse_alignment(temp_aln_name)
        consensus, cons_conservation = compute_consensus(aligned_seqs)
        output_text = format_consensus_output(consensus, cons_conservation)
        consensus_file_path = os.path.join(args.output, "consensus.txt")
        with open(consensus_file_path, 'w') as out_f:
            out_f.write(output_text)
        print(f"Consensus sequence and conservation info written to {consensus_file_path}")
    finally:
        os.remove(temp_input)
        os.remove(temp_aln_name)

    # --- Protein of Interest Mapping ---
    with open(args.protein_of_interest, 'r') as pf:
        prot_seqs = parse_fasta(pf)
    if not prot_seqs:
        raise ValueError(f"No sequences found in the protein of interest file: {args.protein_of_interest}")
    protein_header, protein_seq = prot_seqs[0]
    print(f"Using protein of interest: {protein_header}")

    aligned_consensus, aligned_protein = align_two_sequences(consensus, protein_seq)
    mapped_residues, mapped_pcts = map_conservation(aligned_consensus, aligned_protein, cons_conservation)
    
    print("Mapped protein residues and conservation percentages:")
    for i, (r, p) in enumerate(zip(mapped_residues, mapped_pcts), start=1):
        print(f"Position {i}: {r} -> {p:.1f}%")

    write_masked_outputs_for_threshold(mapped_residues, mapped_pcts, args.output, args.threshold)

if __name__ == "__main__":
    main()
