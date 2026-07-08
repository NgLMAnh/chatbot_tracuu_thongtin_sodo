import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import cv2
import yaml

# Tự động chuyển console output sang UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Disable oneDNN to avoid incompatibility issues on Windows CPU
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from src.pipeline import DocumentPipeline

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def draw_boxes_on_image(image_path, output_path, blocks):
    print(f"Drawing boxes on {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Vẽ từng block
    for block in blocks:
        bbox = block.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Vẽ hình chữ nhật màu đỏ (B, G, R) = (0, 0, 255), độ dày 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Tùy chọn: in thêm block_id hoặc label nhỏ lên trên góc
        label = f"{block.get('block_id', '')} - {block.get('section', 'unknown')}"
        cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Lưu ảnh
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved debug image to: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(base_dir, "configs")
    config_path = os.path.join(configs_dir, "pipeline.yaml")
    
    documents_dir = os.path.join(base_dir, "data", "documents")
    output_base_dir = os.path.join(base_dir, "outputs", "debug_images")

    if not os.path.exists(documents_dir):
        print(f"Directory not found: {documents_dir}")
        return

    print("Khởi tạo Pipeline...")
    config = load_yaml(config_path)
    pipeline = DocumentPipeline(config)

    doc_folders = sorted([d for d in os.listdir(documents_dir) if os.path.isdir(os.path.join(documents_dir, d))])
    
    for doc_id in doc_folders:
        # Skip DOC1 (the original PDF folder) if it's there
        if doc_id == "DOC1":
            continue
            
        doc_path = os.path.join(documents_dir, doc_id)
        image_files = sorted([f for f in os.listdir(doc_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            continue
            
        print(f"\n[{doc_id}] Tìm thấy {len(image_files)} trang ảnh. Đang xử lý...")
        target_dir = os.path.join(output_base_dir, doc_id)
        os.makedirs(target_dir, exist_ok=True)
        
        for img_name in image_files:
            test_image_path = os.path.join(doc_path, img_name)
            output_image_path = os.path.join(target_dir, img_name)
            
            print(f"  -> Đang chạy Pipeline cho {img_name}...")
            try:
                res = pipeline.process_image(test_image_path)
                draw_boxes_on_image(test_image_path, output_image_path, res.get("blocks", []))
            except Exception as e:
                print(f"  -> Lỗi xử lý {img_name}: {e}")

if __name__ == "__main__":
    main()
