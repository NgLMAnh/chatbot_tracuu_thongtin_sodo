import os
import json
import re

def extract_json_from_llm_response(response_text):
    """
    Extracts JSON from LLM response which might contain markdown code blocks.
    """
    # Remove markdown code block markers
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response_text.strip()
        
    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON from LLM: {e}")
        return None

def extract_information(document_id, markdown_content):
    """
    Uses Gemini API to extract structured JSON from the markdown text of a document.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found in environment. Skipping LLM extraction.")
        return {
            "document_id": document_id,
            "error": "GEMINI_API_KEY missing",
            "holder": {"name": None, "id_number": None, "address": None, "birthday": None},
            "land_parcel": {"parcel_number": None, "map_sheet_number": None, "area_m2": None}
        }
        
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("WARNING: 'google-genai' package not installed. Run 'pip install google-genai'.")
        return None

    client = genai.Client(api_key=api_key)
    
    prompt = f"""
Bạn là một chuyên gia trích xuất thông tin từ Giấy chứng nhận quyền sử dụng đất (Sổ đỏ/Sổ hồng).
Dưới đây là nội dung đã được OCR (chuyển thành văn bản) của giấy tờ {document_id}.
Văn bản có thể bị lỗi chính tả hoặc OCR sai đôi chút, hãy cố gắng sửa lỗi logic.

Lưu ý quan trọng:
1. "Chủ sở hữu" hiện tại là người cuối cùng có tên trong phần "Thay đổi về chủ" (nếu có). Nếu không có, đó là người ở Mục I (Chủ sở hữu).
2. CMND/CCCD thường có 9 hoặc 12 số.
3. Diện tích thửa đất (area_m2) là một con số, ví dụ 120.5

Hãy trích xuất thông tin và trả về ĐÚNG định dạng JSON sau, không kèm bất kỳ giải thích nào:
{{
  "document_id": "{document_id}",
  "holder": {{
    "name": "Tên chủ sở hữu hiện tại",
    "id_number": "Số CMND/CCCD",
    "address": "Địa chỉ thường trú",
    "birthday": "Năm sinh (hoặc Ngày tháng năm sinh)"
  }},
  "land_parcel": {{
    "parcel_number": "Số thửa đất",
    "map_sheet_number": "Tờ bản đồ số",
    "area_m2": <số float>
  }}
}}

NỘI DUNG SỔ ĐỎ:
{markdown_content}
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return extract_json_from_llm_response(response.text)
    except Exception as e:
        print(f"LLM API Error: {e}")
        return None
