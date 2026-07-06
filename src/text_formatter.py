def group_boxes_into_lines(ocr_results, y_tolerance_ratio=0.5):
    """
    Groups OCR bounding boxes into lines based on their vertical overlap.
    ocr_results: list of dicts with 'text' and 'bbox' ([x1, y1, x2, y2]).
    """
    # Sort boxes by top y-coordinate
    boxes = sorted(ocr_results, key=lambda b: b['bbox'][1])
    
    lines = []
    current_line = []
    
    for box in boxes:
        if not current_line:
            current_line.append(box)
            continue
            
        # Get the average y1 and y2 of the current line
        line_y1 = sum(b['bbox'][1] for b in current_line) / len(current_line)
        line_y2 = sum(b['bbox'][3] for b in current_line) / len(current_line)
        line_height = line_y2 - line_y1
        
        box_y1, box_y2 = box['bbox'][1], box['bbox'][3]
        box_height = box_y2 - box_y1
        
        # Calculate vertical overlap
        overlap_y1 = max(line_y1, box_y1)
        overlap_y2 = min(line_y2, box_y2)
        overlap_height = max(0, overlap_y2 - overlap_y1)
        
        # If the overlap is greater than 50% of the smaller box's height, they are on the same line
        min_height = min(line_height, box_height)
        if min_height > 0 and overlap_height / min_height >= y_tolerance_ratio:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
            
    if current_line:
        lines.append(current_line)
        
    return lines

def format_as_markdown(page_results_dict):
    """
    Takes a dictionary mapping page names to OCR results and formats them as a single Markdown string.
    """
    md_lines = []
    
    for page_name, ocr_results in page_results_dict.items():
        md_lines.append(f"## {page_name}")
        md_lines.append("")
        
        lines = group_boxes_into_lines(ocr_results)
        
        for line_boxes in lines:
            # Sort boxes in the same line from left to right
            line_boxes = sorted(line_boxes, key=lambda b: b['bbox'][0])
            line_text = " | ".join([b['text'] for b in line_boxes])
            md_lines.append(line_text)
            
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
    return "\n".join(md_lines)
