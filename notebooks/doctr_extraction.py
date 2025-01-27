import os
import json
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from transformers import pipeline

def extract_text_with_bboxes_and_summary(pdf_path, batch_size=5):
    """
    Extract text with bounding boxes and generate a summary in batches.

    Args:
        pdf_path (str): Path to the PDF file.
        batch_size (int): Number of pages to process in each batch.

    Returns:
        dict: A dictionary containing detailed OCR data and a summary of the content.
    """
    # Initialize summary to avoid UnboundLocalError
    summary = None  
    detailed_output = []
    
    try:
        # Load the PDF document
        doc = DocumentFile.from_pdf(pdf_path)
        total_pages = len(doc)

        # Initialize the OCR model and summarizer
        ocr_model = ocr_predictor(pretrained=True)
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

        full_text = ""

        # Process the document in batches
        for start_idx in range(0, total_pages, batch_size):
            batch = doc[start_idx:start_idx + batch_size]  # Get the current batch of pages
            layouts = ocr_model(batch)  # Perform OCR on the batch

            for page_index, page_layout in enumerate(layouts.pages):
                page_data = []
                for block in page_layout.blocks:
                    for line in block.lines:
                        line_data = {
                            "text": ' '.join(word.value for word in line.words),
                            "bounding_box": line.geometry  # Bounding box coordinates
                        }
                        page_data.append(line_data)
                        full_text += line_data["text"] + " "  # Accumulate text for summarization

                detailed_output.append({"page_number": start_idx + page_index + 1, "data": page_data})

            print(f"Page {start_idx + page_index + 1} text: {full_text[:500]}")  # Debug text extraction

        if not full_text:
            print("No text extracted from OCR.")

        # Generate a summary of the accumulated text
        if full_text:
            summary = summarizer(full_text[:2000], max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        else:
            print("No text to summarize.")

    except Exception as e:
        print(f"Failed to process {pdf_path}: {e}")
        raise  # Re-raise to debug further

    return {"ocr_data": detailed_output, "summary": summary}
