import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys

# Tự động chuyển console output sang UTF-8 để không bị lỗi font tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import yaml

import re

# Disable oneDNN to avoid incompatibility issues on Windows CPU
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"


from src.ocr_engine import OCREngine
from src.spatial_rules import normalize_text
from src.extractors import extract_fields
from src.normalizers import normalize_fields
from src.document_merger import merge_pages

def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(base_dir, "configs", "template_a")
    documents_dir = os.path.join(base_dir, "data", "documents")
    outputs_dir = os.path.join(base_dir, "outputs", "predictions")
    
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Load configuration
    try:
        template_config = load_yaml(os.path.join(configs_dir, "template.yaml"))
    except Exception as e:
        print(f"Error loading template config: {e}")
        sys.exit(1)
        
    print(f"Loaded template: {template_config.get('template_name')} (ID: {template_config.get('template_id')})")
    
    # 2. Find document folders
    if not os.path.exists(documents_dir):
        print(f"Error: Documents directory '{documents_dir}' does not exist.")
        return
        
    doc_folders = sorted([d for d in os.listdir(documents_dir) if os.path.isdir(os.path.join(documents_dir, d))])
    
    if not doc_folders:
        print("No document folders found in data/documents.")
        return
        
    print(f"Found {len(doc_folders)} documents to process: {', '.join(doc_folders)}")
    
    # 3. Initialize OCREngine
    print("\nInitializing OCR Engine (PaddleOCR)... This might take a few seconds.")
    try:
        # Using CPU (gpu=False) for compatibility
        ocr_engine = OCREngine(languages=['vi'], gpu=False)
    except Exception as e:
        print(f"Error initializing OCR Engine: {e}")
        sys.exit(1)
        
    print("OCR Engine ready.\n")
    print("=" * 80)
    print(" PROCESSING DOCUMENTS")
    print("=" * 80)
    
    # 4. Process each document folder
    for doc_id in doc_folders:
        doc_path = os.path.join(documents_dir, doc_id)
        image_files = sorted([f for f in os.listdir(doc_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"Skipping {doc_id}: No image files found.")
            continue
            
        print(f"\nProcessing document: {doc_id} ({len(image_files)} pages)")
        page_results_dict = {}
        
        for img_name in image_files:
            img_path = os.path.join(doc_path, img_name)
            print(f"  - Running OCR on: {img_name}...")
            
            try:
                # Perform OCR
                ocr_results = ocr_engine.run_ocr(img_path)
                page_results_dict[img_name] = ocr_results
            except Exception as e:
                import traceback
                print(f"    -> Error processing page {img_name}: {e}")
                traceback.print_exc()
                
        # 5. Format as Markdown
        from src.text_formatter import format_as_markdown
        md_text = format_as_markdown(page_results_dict)
        
        # Save Markdown
        md_dir = os.path.join(base_dir, "outputs", "markdowns")
        os.makedirs(md_dir, exist_ok=True)
        md_file = os.path.join(md_dir, f"{doc_id}.md")
        try:
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"    -> Saved markdown to: {md_file}")
        except Exception as e:
            print(f"    -> Error saving markdown for {doc_id}: {e}")
        
        # 6. Extract using LLM (RAG)
        print("  - Extracting information using LLM...")
        from src.llm_extractor import extract_information
        doc_json = extract_information(doc_id, md_text)
        
        if doc_json:
            # Save output JSON
            output_file = os.path.join(outputs_dir, f"{doc_id}.json")
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(doc_json, f, ensure_ascii=False, indent=2)
                print(f"\nSUCCESS: Document {doc_id} result saved to: {output_file}")
                print(json.dumps(doc_json, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"Error saving result for {doc_id}: {e}")
            
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
