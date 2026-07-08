import sys
# Tự động chuyển console output sang UTF-8 để không bị lỗi font tiếng Việt trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from src.vector_store import load_vector_store

def main():
    print("==================================================")
    print("🤖 CHATBOT HỎI ĐÁP SỔ ĐỎ (CLI MODE)")
    print("==================================================")
    
    print("⏳ Đang tải cơ sở dữ liệu Vector (ChromaDB)...")
    try:
        vector_store = load_vector_store()
    except Exception as e:
        print(f"❌ Lỗi tải Vector DB: {e}. Bạn đã chạy bước Embedding (test_embedding.py) chưa?")
        return
        
    print("⏳ Đang khởi tạo mô hình AI (Ollama qwen2.5:7b)...")
    try:
        llm = Ollama(model="qwen2.5:7b", temperature=0.0)
    except Exception as e:
        print(f"Lỗi tải Ollama: {e}")
        return

    prompt_template = """Bạn là một trợ lý thông minh chuyên giải đáp các câu hỏi về thông tin Giấy chứng nhận quyền sử dụng đất (Sổ đỏ/Sổ hồng).
Dựa vào các NỘI DUNG TRÍCH XUẤT từ Sổ đỏ dưới đây, hãy trả lời câu hỏi của người dùng bằng tiếng Việt một cách ngắn gọn, chính xác.
Nếu thông tin không có trong NỘI DUNG TRÍCH XUẤT, hãy trả lời trung thực là "Tôi không tìm thấy thông tin này trong tài liệu hiện tại."

ĐẶC BIỆT LƯU Ý:
1. Nếu có thông tin từ nhiều nguồn tài liệu khác nhau, hãy phân loại câu trả lời rõ ràng theo từng tài liệu.
2. ƯU TIÊN SỐ 1: Nếu trong NỘI DUNG TRÍCH XUẤT có phần "# document_summary" hoặc "extracted_fields" (chứa các trường như chu_so_huu, cmnd_cccd, dia_chi, dien_tich_m2...), BẮT BUỘC phải sử dụng thông tin ở phần này vì đây là dữ liệu chính xác nhất.
3. Người ký giấy tờ (Chủ tịch, Phó Chủ tịch, Ủy viên, Chủ hộ, Giám đốc) KHÔNG PHẢI là chủ sở hữu.
4. Nếu người dùng hỏi về Chủ sở hữu và không có phần summary, hãy chú ý tìm người có các từ "Ông:", "Bà:" kèm theo Năm sinh và CMND.
5. LUÔN trả lời ĐÚNG TRỌNG TÂM câu hỏi. Ví dụ: Hỏi địa chỉ thì CHỈ trả lời địa chỉ, không trả lời tên chủ sở hữu.
6. LƯU Ý TỪ VỰNG: "cmnd_cccd" chính là số Chứng minh nhân dân (CMND) hoặc Căn cước công dân (CCCD). "chu_so_huu" là Chủ sở hữu.

VÍ DỤ CÁCH ĐỌC:
Văn bản trích xuất có đoạn:
"# document_summary
- chu_so_huu: NGUYỄN VĂN A
- parcel_number: 123"
Và một đoạn khác:
"Trần Văn B | KT. Chủ tịch"
Câu hỏi: "Ai là chủ sở hữu?"
Cách tư duy: Có phần "# document_summary" ghi rõ holder_name là Nguyễn Văn A. Trần Văn B là người ký giấy.
Câu trả lời đúng: "Chủ sở hữu là Ông Nguyễn Văn A."

NỘI DUNG TRÍCH XUẤT:
{context}

CÂU HỎI CỦA NGƯỜI DÙNG: {question}

TRẢ LỜI:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    print("\n✅ Khởi tạo thành công! (Gõ 'quit' hoặc 'exit' để thoát)")
    
    while True:
        try:
            print("\n--------------------------------------------------")
            user_input = input("🗣️ Bạn: ")
            if user_input.lower().strip() in ['quit', 'exit', 'q']:
                print("Tạm biệt!")
                break
            if not user_input.strip():
                continue
                
            print("⏳ Đang tìm kiếm thông tin...")
            # Lấy 4 chunks liên quan nhất
            results = vector_store.similarity_search(user_input, k=4)
            
            context_chunks = []
            for doc in results:
                doc_id = doc.metadata.get("document_id", "Không rõ")
                page = doc.metadata.get("page", "Không rõ")
                context_chunks.append(f"[Nguồn: {doc_id} - Trang: {page}]:\n{doc.page_content}")
                
            unique_context = list(set(context_chunks))
            combined_context = "\n\n---\n\n".join(unique_context)
            
            print("\n[DEBUG] RAG đã tìm thấy các đoạn văn bản sau:")
            print(combined_context)
            print("--------------------------------------------------\n")
            
            print("⏳ Đang suy nghĩ trả lời...")
            response = chain.invoke({
                "context": combined_context,
                "question": user_input
            })
            
            print(f"\n🤖 Chatbot:\n{response.strip()}")
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
