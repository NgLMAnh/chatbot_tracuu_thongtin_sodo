import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store(chunks, persist_directory="outputs/chroma_db"):
    """
    Nhận danh sách các chunks và đưa vào Vector Database (Chroma).
    Sử dụng mô hình Embedding mã nguồn mở tiếng Việt hoặc đa ngôn ngữ.
    """
    if not chunks:
        print("Không có chunks nào để đưa vào DB.")
        return None
        
    import shutil
    if os.path.exists(persist_directory):
        print(f"Đang xóa database cũ tại {persist_directory}...")
        shutil.rmtree(persist_directory)
        
    print(f"Đang load embedding model...")
    # Sử dụng BGE-M3: Mô hình đa ngôn ngữ siêu việt, tối ưu RAG
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # Chuyển đổi định dạng list[dict] sang list[str] và list[dict] metadatas cho Chroma
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    print(f"Đang embedding {len(texts)} chunks và lưu vào {persist_directory}...")
    
    # Khởi tạo và lưu vào Chroma DB
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    
    print("Hoàn tất!")
    return vector_store

def load_vector_store(persist_directory="outputs/chroma_db"):
    """Load Chroma DB đã lưu."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

