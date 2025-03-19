import csv
import yaml
import argparse
import os
import string
import re

# Marker class for strings that must always be single-quoted in YAML
class SingleQuotedString(str):
    pass

def single_quoted_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

# A list that should always be dumped in flow style, e.g. [A, B].
class FlowList(list):
    pass

def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Custom YAML dumper with increased indent for better readability
class IndentDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, indentless=False)

IndentDumper.add_representer(FlowList, flow_list_representer)
IndentDumper.add_representer(SingleQuotedString, single_quoted_str_representer)

def generate_id_list(start_index, count):
    return [string.ascii_uppercase[i] for i in range(start_index, start_index + count)]

def sanitize_filename(name):
    # Replace any sequence of non-word characters with an underscore
    return re.sub(r'\W+', '_', name)

def csv_to_yaml(csv_filename):
    if not os.path.isfile(csv_filename):
        print("Error: File does not exist:", csv_filename)
        return

    with open(csv_filename, mode="r", encoding="utf-8") as csv_file:
        reader = list(csv.DictReader(csv_file))
        print("Found", len(reader), "data rows in:", csv_filename)
        if not reader:
            print("No data rows found in the CSV.")
            return

    # Use the same directory as the CSV file for output
    csv_directory = os.path.dirname(csv_filename)

    # Process each CSV row (each represents a separate docking simulation)
    for i, row in enumerate(reader, start=1):
        # Get protein data
        protein_name = row.get("protein_name", f"user_{i}").strip()
        protein_sequence = row.get("protein_sequence", "").strip()
        protein_count = int(row.get("protein_count", 1) or 1)

        # Assign protein chain IDs starting at A
        protein_ids = FlowList(generate_id_list(0, protein_count))
        # Ligand chain letters follow immediately after protein chains.
        ligand_start_index = protein_count

        # Build YAML structure with the protein entry first.
        yaml_data = {
            "version": 1,
            "sequences": []
        }
        yaml_data["sequences"].append({
            "protein": {
                "id": protein_ids,
                "sequence": protein_sequence
            }
        })

        # Process ligand columns and collect ligand names for file naming.
        ligand_names = []
        current_ligand_index = ligand_start_index
        # Iterate over a fixed range (adjust 101 as needed)
        for j in range(1, 101):
            ligand_count_key = f"ligand{j}_count"
            ligand_sequence_key = f"ligand{j}_sequence"
            ligand_smiles_or_ccd_key = f"ligand{j}_smiles_or_ccd"
            ligand_name_key = f"ligand{j}_name"

            # If the ligand column doesn't exist, skip to next index.
            if ligand_count_key not in row:
                continue

            # If the column exists but is empty, skip this ligand.
            if not row[ligand_count_key].strip():
                continue

            # Process the ligand.
            ligand_count = int(row[ligand_count_key].strip())
            ligand_ids = FlowList(generate_id_list(current_ligand_index, ligand_count))
            current_ligand_index += ligand_count

            ligand_name_val = row.get(ligand_name_key, "").strip()
            if not ligand_name_val:
                print(f"Warning: Missing ligand name for row {i}, ligand {j}. Skipping this ligand.")
                continue
            ligand_names.append(ligand_name_val)

            # Ensure ligand sequence and mode exist.
            if not (row.get(ligand_sequence_key, "").strip() and row.get(ligand_smiles_or_ccd_key, "").strip()):
                print(f"Warning: Incomplete data for row {i}, ligand {j}. Skipping.")
                continue

            ligand_mode = row[ligand_smiles_or_ccd_key].strip().lower()
            if ligand_mode == "smiles":
                raw_smiles = row[ligand_sequence_key].strip()
                ligand_entry = {
                    "ligand": {
                        "id": ligand_ids,
                        "smiles": SingleQuotedString(raw_smiles)
                    }
                }
            elif ligand_mode == "ccd":
                raw_ccd = row[ligand_sequence_key].strip()
                ligand_entry = {
                    "ligand": {
                        "id": ligand_ids,
                        "ccd": SingleQuotedString(raw_ccd)
                    }
                }
            else:
                print(f"Warning: Unknown ligand mode '{ligand_mode}' for row {i}, ligand {j}. Skipping.")
                continue

            yaml_data["sequences"].append(ligand_entry)

        # Build the output file name using the protein name and all ligand names.
        if ligand_names:
            ligand_part = "_".join(sanitize_filename(name) for name in ligand_names)
        else:
            ligand_part = "no_ligand"
        sanitized_protein_name = sanitize_filename(protein_name)
        yaml_filename = os.path.join(csv_directory, f"{sanitized_protein_name}_docked_with_{ligand_part}.yaml")

        with open(yaml_filename, mode="w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_data,
                yaml_file,
                Dumper=IndentDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                indent=2
            )

        print("Created YAML file:", yaml_filename)

    print("Done converting CSV rows to YAML.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV rows into structured YAML files.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    args = parser.parse_args()
    csv_to_yaml(args.csv_file)
