import json
import os
import pandas as pd

def json_to_pandas(input_json_path, output_csv_path):
    """
    Converts a JSON file with 'original_text_chunks' and 'summaries' fields into a CSV file.

    Parameters:
    - input_json_path: Path to the input JSON file.
    - output_csv_path: Path to save the resulting CSV file.
    """
    # Load JSON data
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract relevant fields
    original_chunks = data.get("original_text_chunks", [])
    summaries = data.get("summaries", [])

    # Ensure the data is consistent
    if len(original_chunks) != len(summaries):
        print(f"Error: Mismatch in chunks and summaries for {input_json_path}")
        return

    # Create a DataFrame
    df = pd.DataFrame({
        "Chunk Number": range(1, len(original_chunks) + 1),
        "Original Text Chunk": original_chunks,
        "Summary": summaries
    })

    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Converted {input_json_path} to {output_csv_path}")

def convert_folder_json_to_csv(input_folder, output_folder):
    """
    Converts all JSON files in a specified folder to CSV files.

    Parameters:
    - input_folder: Path to the folder containing JSON files.
    - output_folder: Path to save resulting CSV files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):  # Only process JSON files
            input_json_path = os.path.join(input_folder, filename)
            output_csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")

            print(f"Processing file: {input_json_path}")
            json_to_pandas(input_json_path, output_csv_path)

# Example usage
if __name__ == "__main__":
    input_folder = 'output'  # Replace with the folder containing your JSON files
    output_folder = 'csv_outputs'  # Folder to save the converted CSV files
    convert_folder_json_to_csv(input_folder, output_folder)
