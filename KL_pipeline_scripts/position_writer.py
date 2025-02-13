#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
import os
from collections import Counter

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
    Parse a FASTA file given a file handle.
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

def combine_fasta_files(fasta_filepaths):
    """
    Given a list of FASTA file paths, read and combine all sequences.
    Returns a list of (header, sequence) tuples.
    """
    combined = []
    for filepath in fasta_filepaths:
        with open(filepath, 'r') as f:
            seqs = parse_fasta(f)
            if not seqs:
                print(f"Warning: No sequences found in {filepath}")
            combined.extend(seqs)
    return combined

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
    Run Clustal Omega to perform multiple sequence alignment.
    The output alignment will be in FASTA format.
    """
    # The --force flag will overwrite the output file if it exists.
    # The --outfmt=fa option sets the output format to FASTA.
    cmd = ["clustalo", "-i", input_fasta, "-o", output_aln, "--force", "--outfmt=fa"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Clustal Omega failed: {e}")

def parse_alignment(aln_filepath):
    """
    Parse an alignment file (in FASTA format) into a list of sequences.
    Returns a list of aligned sequence strings.
    """
    aligned_seqs = []
    with open(aln_filepath, 'r') as f:
        for header, seq in parse_fasta(f):
            aligned_seqs.append(seq)
    if not aligned_seqs:
        raise ValueError("No sequences found in the alignment.")
    # Ensure all sequences are the same length
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
    
    # Iterate column-by-column
    for pos in range(aln_length):
        # Collect all characters in this column.
        column = [seq[pos] for seq in aligned_seqs]
        counter = Counter(column)
        # Ignore gap characters for consensus determination.
        non_gap = {res: cnt for res, cnt in counter.items() if res != '-'}
        if non_gap:
            # Choose the amino acid with the highest count.
            res, cnt = max(non_gap.items(), key=lambda x: x[1])
        else:
            # If every sequence has a gap at this position, record as gap.
            res = '-'
            cnt = counter.get('-', 0)
        percent = (cnt / num_seqs) * 100
        consensus.append(res)
        conservation.append(percent)
    return ''.join(consensus), conservation

def format_consensus_output(consensus, conservation):
    """
    Format the consensus sequence and per-position conservation percentages.
    Returns a multi-line string.
    """
    output_lines = []
    # First line: full consensus sequence
    output_lines.append("Consensus sequence:")
    output_lines.append(consensus)
    output_lines.append("")  # blank line for separation
    output_lines.append("Position-by-position conservation:")
    
    # For each position, show the consensus residue (as three-letter code if possible) and conservation
    for i, (res, perc) in enumerate(zip(consensus, conservation), start=1):
        # Convert residue to three-letter code if available.
        # If the residue is a gap or not found in the mapping, use the residue as is.
        res_display = ONE_TO_THREE.get(res, res)
        # Format the percentage as an integer (rounded).
        output_lines.append(f"Position {i}: {res_display} : {int(round(perc))}%")
    return "\n".join(output_lines)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a consensus sequence from multiple FASTA files using Clustal Omega for MSA."
    )
    parser.add_argument("--fasta", nargs='+', required=True,
                        help="Path(s) to FASTA file(s).")
    parser.add_argument("--output", required=True,
                        help="Path to the output folder where the consensus file will be written.")
    args = parser.parse_args()

    # Combine sequences from all input FASTA files.
    sequences = combine_fasta_files(args.fasta)
    if not sequences:
        raise ValueError("No sequences found in the provided FASTA files.")

    # Write the combined sequences to a temporary FASTA file.
    temp_input = write_temp_fasta(sequences)

    # Create a temporary file for the alignment output.
    temp_aln_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".aln")
    temp_aln_file.close()  # We only need its name; clustalo will write to it.
    temp_aln_name = temp_aln_file.name

    try:
        # Run Clustal Omega to perform the multiple sequence alignment.
        run_clustalo(temp_input, temp_aln_name)

        # Parse the alignment.
        aligned_seqs = parse_alignment(temp_aln_name)

        # Compute the consensus sequence and per-position conservation.
        consensus, conservation = compute_consensus(aligned_seqs)

        # Format the output.
        output_text = format_consensus_output(consensus, conservation)

        # Ensure the output folder exists.
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        # Define the consensus file path inside the output folder.
        consensus_file_path = os.path.join(args.output, "consensus.txt")

        # Write the output to the consensus file.
        with open(consensus_file_path, 'w') as out_f:
            out_f.write(output_text)
        print(f"Consensus sequence and conservation information written to {consensus_file_path}")
    finally:
        # Clean up temporary files.
        os.remove(temp_input)
        os.remove(temp_aln_name)

if __name__ == "__main__":
    main()
