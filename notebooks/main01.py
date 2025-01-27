import json
import os
from doctr_extraction import extract_text_with_bboxes_and_summary

def main():
    # Define input and output paths using os.path.expanduser
    pdf_folder_path = os.path.expanduser('~/Textra-edu/data/pdf')
    output_folder_path = os.path.expanduser('~/Textra-edu/data/raw')
    
    # Verify PDF folder exists
    if not os.path.exists(pdf_folder_path):
        print(f"Error: PDF folder does not exist: {pdf_folder_path}")
        return
    
    # Create output folder
    os.makedirs(output_folder_path, exist_ok=True)
    
    # List PDFs in the folder
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder_path}")
        return
    
    # Process each PDF
    print(f"Found {len(pdf_files)} PDF files. Processing...")
    text_data_by_pdf = {}
    
    for pdf_name in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_name)
        try:
            # Extract text from individual PDF
            pdf_data = extract_text_with_bboxes_and_summary(pdf_path)
            text_data_by_pdf[pdf_name] = pdf_data
            
            # Save results
            json_filename = os.path.splitext(pdf_name)[0] + '.json'
            output_file_path = os.path.join(output_folder_path, json_filename)
            
            with open(output_file_path, 'w', encoding='utf-8') as file:
                json.dump(pdf_data, file, ensure_ascii=False, indent=4)
            
            print(f"Processed {pdf_name}: Summary and detailed OCR saved to {output_file_path}")
        
        except Exception as e:
            print(f"Failed to process {pdf_name}: {e}")

if __name__ == "__main__":
    main()