# NER OCR API

API OCR tài liệu bằng Google Document AI, tự phân loại thành `Hợp đồng` hoặc `Tài liệu`, rồi trích xuất thông tin theo 2 màn nghiệp vụ.

## Chạy local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở demo UI: http://localhost:8000/demo

Mở demo UI có bounding box: http://localhost:8000/demo-layout

## API chính

```bash
curl -X POST "http://localhost:8000/api/ocr/extract" \
  -F "file=@/path/to/document.pdf"
```

API OCR + LLM + layout bounding boxes:

```bash
curl -X POST "http://localhost:8000/api/ocr/extract-layout" \
  -F "file=@/path/to/document.pdf"
```

Test riêng tầng LLM information extraction bằng OCR text/raw text:

```bash
curl -X POST "http://localhost:8000/api/llm/extract" \
  -H "Content-Type: application/json" \
  -d '{"text":"HỢP ĐỒNG THI CÔNG... Số: 12/2025/HĐ-XD... Giá trị hợp đồng: 1.200.000.000 đồng"}'
```

Response gồm:

- `ocr.text`: text OCR đầy đủ.
- `extraction.document_type`: `contract` hoặc `document`.
- `extraction.screen`: `contract` hoặc `work_detail`.
- `extraction.fields`: các field của màn tương ứng.
- `extraction.generic_extraction`: thông tin mặc định khi tài liệu thiếu field nghiệp vụ.
- `taxonomies.contract_forms`: danh sách hình thức hợp đồng tham chiếu.
- `taxonomies.contractor_groups`: danh sách nhóm nhà thầu tham chiếu.
- `layout.segments`: các đoạn OCR có bounding box normalized theo trang.
- `extraction.fields[*].box`: bounding box đã map về field trong form.

Nếu có `OPENAI_API_KEY`, hệ thống dùng OpenAI để trích xuất. Nếu chưa cấu hình key, API vẫn trả kết quả heuristic để kiểm thử OCR và UI.
