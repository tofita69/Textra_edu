import json
import os
from transformers import pipeline

def summarize_text(input_json_path, output_json_path, chunk_size=5):
    # Load the summarization model
    summarizer = pipeline("summarization", model="google/pegasus-xsum")

    # Read JSON data
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Structure: { "ocr_data": [{ "page_number": ..., "data": [...] }] }

    # Prepare text for summarization
    combined_text = []
    for page in data.get("ocr_data", []):
        page_text = [item.get("text", "") for item in page.get("data", [])]
        combined_text.extend(page_text)

    # Split text into chunks
    chunks = [
        " ".join(combined_text[i:i + chunk_size])
        for i in range(0, len(combined_text), chunk_size)
    ]

    # Summarize each chunk
    summaries = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            summaries.append(f"Chunk {idx + 1}: No text available")
            continue

        # Calculate dynamic max_length based on chunk length
        input_length = len(chunk.split())  # Number of words in the chunk
        max_length = min(60, max(12, input_length // 2))  # Ensure summaries are shorter but not too short

        try:
            summary = summarizer(chunk[:1024], max_length=max_length, min_length=10, do_sample=False)[0]["summary_text"]
            summaries.append(f"Chunk {idx + 1}: {summary}")
        except Exception as e:
            summaries.append(f"Chunk {idx + 1}: Summarization failed ({e})")

    # Save summarized data to a new JSON file
    output_data = {
        "original_text_chunks": chunks,
        "summaries": summaries
    }
    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    print(f"Summarization complete. Output saved to {output_json_path}")

def summarize_folder(input_folder, output_folder, chunk_size=5):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_json_path = os.path.join(input_folder, filename)
            output_json_path = os.path.join(output_folder, f"summarized_{filename}")
            print(f"Processing file: {input_json_path}")
            summarize_text(input_json_path, output_json_path, chunk_size)

# Example usage
input_folder = 'data/raw'  # Folder containing input JSON files
output_folder = 'output'  # Folder to save output JSON files
summarize_folder(input_folder, output_folder, chunk_size=5)