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

Mở demo UI work_detail: http://localhost:8000/demo-work-detail

Alias demo chính: http://localhost:8000/demo

Demo UI có dropdown dự án `executing`, lấy từ `GET /api/work-detail/projects` để kiểm tra/mapping project trước hoặc sau khi OCR.

Mở demo UI có bounding box: http://localhost:8000/demo-layout

## API chính

```bash
curl -X POST "http://localhost:8000/api/ocr/extract" \
  -F "file=@/path/to/document.pdf"
```

API OCR + Gemini fallback + layout bounding boxes:

```bash
curl -X POST "http://localhost:8000/api/ocr/extract-layout" \
  -F "file=@/path/to/document.pdf"
```

Test riêng tầng information extraction bằng OCR text/raw text:

```bash
curl -X POST "http://localhost:8000/api/llm/extract" \
  -H "Content-Type: application/json" \
  -d '{"text":"HỢP ĐỒNG THI CÔNG... Số: 12/2025/HĐ-XD... Giá trị hợp đồng: 1.200.000.000 đồng"}'
```

Lấy danh sách project `executing` cho dropdown demo:

```bash
curl -X GET "http://localhost:8000/api/work-detail/projects"
```

Response gồm:

- `ocr.text`: text OCR đầy đủ.
- `extraction.document_type`: `contract` hoặc `document`.
- `extraction.screen`: `contract` hoặc `work_detail`.
- `extraction.fields`: các field của màn tương ứng.
- `extraction.fields.title`: title/tên tờ trình dùng làm tín hiệu match dự án/công việc.
- `extraction.generic_extraction`: intent/entities rộng để matching, gồm project candidates, task candidates, work items, gói thầu, task keywords, dates, monetary amounts.
- `extraction.entities`: entity chi tiết do Gemini enrich nếu có credential.
- `extraction.work_detail_match`: kết quả match dự án `executing` và công việc trong dự án qua Agribank internal API.
- `extraction.work_detail_output`: output phẳng để backend fill tự động, gồm số văn bản, ngày/thời gian, giá trị duyệt/trình, issuer, notes, title, mã/tên dự án, task id/name và workflow step nếu match được.
- `extraction.pipeline`: `local`, `local_with_llm_fallback`, hoặc `llm`.
- `extraction.local_confidence`: điểm tin cậy local tổng hợp.
- `extraction.llm_fallback_used`: `true` nếu hệ thống đã gọi Gemini để enrich/fallback extraction.
- `extraction.llm_entity_extraction_used`: `true` nếu Gemini được dùng ngay từ bước intent/entities.
- `extraction.llm_extraction_mode`: `entity_extraction` hoặc `fallback`.
- `extraction.needs_review`: `true` nếu local/matching chưa đủ chắc để auto-fill.
- `extraction.work_detail_match.best_candidates`: top 3 project/task candidates khi match chưa chắc.
- `taxonomies.contract_forms`: danh sách hình thức hợp đồng tham chiếu.
- `taxonomies.contractor_groups`: danh sách nhóm nhà thầu tham chiếu.
- `layout.segments`: các đoạn OCR có bounding box normalized theo trang.
- `extraction.fields[*].box`: bounding box đã map về field trong form.

Mặc định hệ thống chạy extraction theo hướng local baseline + Gemini entity enrichment: rule/regex xử lý trước để có guardrail, sau đó nếu có `GEMINI_API_KEY` hoặc Vertex AI credential qua `GOOGLE_APPLICATION_CREDENTIALS`, Gemini đọc OCR đã clean để trích xuất intent/entities rộng hơn. Matching dự án/công việc vẫn ưu tiên local hybrid matching và không gửi danh sách 300 tasks lên LLM. Nếu Gemini lỗi credential/model/quota, API vẫn trả kết quả local kèm `llm_fallback_error` thay vì văng 500.

Với hợp đồng, service áp dụng thêm rule từ tài liệu mô tả bóc tách: ưu tiên điều khoản chính trong hợp đồng, không tự suy diễn field thiếu, chuẩn hóa tiền về số VND, bắt thời gian thực hiện theo số ngày, nhận diện giá hợp đồng/giá gói thầu/VAT/nhà thầu/bảo lãnh, và coi `work_name` là field mapping để phục vụ match task/subtask.

Để bật Gemini fallback, dùng một trong hai cách:

```bash
# Cách 1: Gemini API key trực tiếp
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=90
LLM_ENTITY_EXTRACTION_ENABLED=true
LLM_ENTITY_MAX_CHARS=30000

# Cách 2: Vertex AI Gemini qua service account
GOOGLE_APPLICATION_CREDENTIALS=credentials/credentials.json
GEMINI_VERTEX_PROJECT_ID=... # có thể bỏ trống để dùng GOOGLE_AI_PROJECT_ID
GEMINI_VERTEX_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash
```

Để bật match dự án/công việc cho `work_detail`, cấu hình:

```bash
AGRIBANK_API_BASE_URL=https://agribank-be.opa.ai.vn/api/v1
AGRIBANK_API_KEY=...
LOCAL_EXTRACTION_ENABLED=true
LLM_FALLBACK_ENABLED=true
LLM_ENTITY_EXTRACTION_ENABLED=true
PROJECT_MATCH_THRESHOLD=0.78
TASK_MATCH_THRESHOLD=0.55
```
