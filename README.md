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

Fast API cho PDF (OCR/rule/box trả trước, LLM và project/task matching chạy ở worker):

```bash
curl -i -X POST \
  "http://localhost:8000/api/v2/extractions/file?project=agribank&type=document&include_layout=true" \
  -F "file=@/path/to/document.pdf"

# Poll URL trong header Location; 202 khi đang enrich, 200 khi hoàn tất.
curl -i "http://localhost:8000/api/v2/extractions/<job_id>"
```

V2 dùng Redis Stream và cần worker riêng. Job được giữ 24 giờ; worker chỉ ACK sau khi đã lưu revision cuối:

```bash
redis-server
python -m app.worker
```

Cloud Vision cần được bật trong Google Cloud trước khi chuyển sang primary:

```bash
gcloud services enable vision.googleapis.com --project "$GOOGLE_AI_PROJECT_ID"
OCR_PROVIDER_MODE=vision_shadow  # kiểm chứng trước
# vision_canary -> vision_primary sau khi đạt quality/fallback gate
```

Endpoint V1 cũ vẫn chạy đồng bộ và giữ nguyên payload.

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

Mặc định hệ thống chạy `LLM_EXECUTION_MODE=adaptive`: rule/regex xử lý trước và chỉ gọi Gemini khi field bắt buộc còn thiếu hoặc confidence thấp. Có thể dùng `always` để luôn gọi, hoặc `off` để chỉ chạy local. `LLM_FALLBACK_ENABLED=false` vẫn là công tắc tắt tương thích ngược. Matching dự án/công việc ưu tiên local hybrid matching và không gửi danh sách 300 tasks lên LLM. Nếu Gemini lỗi credential/model/quota, API vẫn trả kết quả local kèm `llm_fallback_error` thay vì văng 500.

Với hợp đồng, service áp dụng thêm rule từ tài liệu mô tả bóc tách: ưu tiên điều khoản chính trong hợp đồng, không tự suy diễn field thiếu, chuẩn hóa tiền về số VND, bắt thời gian thực hiện theo số ngày, nhận diện giá hợp đồng/giá gói thầu/VAT/nhà thầu/bảo lãnh, và coi `work_name` là field mapping để phục vụ match task/subtask.

Để bật Gemini fallback, dùng một trong hai cách:

```bash
# Cách 1: Gemini API key trực tiếp
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=90
LLM_EXECUTION_MODE=adaptive
LLM_ADAPTIVE_CONFIDENCE_THRESHOLD=0.72
LLM_ENTITY_EXTRACTION_ENABLED=true
LLM_PROMPT_MODE=delta
LLM_DOCUMENT_MAX_OCR_CHARS=5000
LLM_CONTRACT_MAX_OCR_CHARS=10000

# Cách 2: Vertex AI Gemini qua service account
GOOGLE_APPLICATION_CREDENTIALS=credentials/credentials.json
GEMINI_VERTEX_PROJECT_ID=... # có thể bỏ trống để dùng GOOGLE_AI_PROJECT_ID
GEMINI_VERTEX_LOCATION=global
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
LLM_MATCH_TIEBREAKER_MAX_GAP=0.15
LLM_MATCH_TIEBREAKER_TIMEOUT_SECONDS=15
```

Mọi request có header `X-Request-ID` và server ghi structured timing log cho từng stage. Thêm query `debug_timing=true` vào extraction endpoint để nhận `timings` trong JSON cùng header `Server-Timing`. OCR Document AI dùng RPC timeout/retry deadline riêng; PDF trên giới hạn trang được xử lý song song nhưng vẫn ghép theo đúng thứ tự:

```bash
DOCUMENT_AI_RPC_TIMEOUT_SECONDS=60
DOCUMENT_AI_RETRY_DEADLINE_SECONDS=75
OCR_MAX_PARALLEL_PARTS=2
PERF_SLOW_STAGE_MS=3000
```

## Rollout Cloud Vision (V2 fast path)

Quy trình promote Vision cho path V2 (`/api/v2/extractions/file`):

1. Enable API + cấp quyền (cần admin GCP project):

   ```bash
   gcloud services enable vision.googleapis.com documentai.googleapis.com aiplatform.googleapis.com \
     --project "$GOOGLE_AI_PROJECT_ID"
   # Service account cần roles/serviceusage.serviceUsageConsumer để header x-goog-user-project hoạt động.
   ```

2. Smoke test credentials (gồm cả Vision):

   ```bash
   python test_credential.py
   ```

3. Bật worker + API rồi chạy shadow để so sánh similarity OCR Vision vs Document AI:

   ```bash
   redis-server &
   OCR_PROVIDER_MODE=vision_shadow python -m app.worker &
   bash run_api.sh
   # Grep log: event=vision_shadow_complete + text_similarity >= 0.95
   ```

4. Chuyển sang canary 10% rồi promote:

   ```bash
   OCR_PROVIDER_MODE=vision_canary VISION_CANARY_PERCENT=10  # quan sát 30 phút
   OCR_PROVIDER_MODE=vision_primary GEMINI_VERTEX_LOCATION=global
   ```

5. Tuning Vision (đã tối ưu theo benchmark 13 file thực):

   ```bash
   VISION_OCR_TIMEOUT_SECONDS=12      # 4.2 quá ngắn cho file >5MB; tăng lên 12
   VISION_MAX_PAGES_PER_REQUEST=5     # hạn mức cứng của Vision files:annotate
   VISION_MAX_PARALLEL_PARTS=8        # parallel cao cho file nhiều chunk
   VISION_MIN_WORD_CONFIDENCE=0.70
   VISION_MIN_WORD_CONFIDENCE=0.70   # fallback Document AI khi confidence trung bình dưới ngưỡng
   VISION_QUALITY_GATE_ENABLED=true   # đặt false để tắt toàn bộ gate
   VISION_CIRCUIT_FAILURE_THRESHOLD=2
   VISION_CIRCUIT_BREAKER_SECONDS=300
   # Cảnh báo khi tỉ lệ fallback vượt ngưỡng (event=vision_unhealthy):
   VISION_FALLBACK_ALERT_WINDOW_S=300
   VISION_FALLBACK_ALERT_THRESHOLD=0.2
   VISION_FALLBACK_ALERT_MIN_SAMPLES=20
   VISION_FALLBACK_ALERT_COOLDOWN_S=60
   ```

   Quality gate chỉ fallback Document AI khi: (a) page count Vision trả về không khớp số trang PDF gốc, hoặc (b) `mean_word_confidence < VISION_MIN_WORD_CONFIDENCE`. Trang trống/ảnh không OCR được (bìa, mặt sau scan) **luôn được chấp nhận** — chỉ ghi nhận tại `result.ocr.quality.short_pages` để quan sát.

## Async OCR cho file lớn (deferred OCR mode)

Mặc định V2 endpoint chạy OCR đồng bộ (khách đợi ~3–7s tuỳ size). Với file >5MB, có thể chuyển OCR sang worker để API trả `202` ngay (~30ms):

```bash
LARGE_FILE_THRESHOLD_BYTES=5242880   # >= 5MB → chạy deferred OCR (0 = tắt, default)
```

Flow khi bật:

1. Client `POST /api/v2/extractions/file` với file lớn → API ngắt OCR, lưu file vào `uploads/`, enqueue với `status="ocr_pending"`, trả `202` + `Location`.
2. Worker pick up job, chạy OCR (Vision → fallback Document AI nếu cần), update record lên `revision=1` (`status="enriching"`).
3. Worker tiếp tục LLM enrichment + matching, update `revision=2` (`status="completed"`). Tự xóa file upload.
4. Client poll `GET /api/v2/extractions/{job_id}` cho đến khi `status ∈ {completed, completed_with_warnings, failed}`.

State machine:

```
ocr_pending  (rev 0, no OCR data)
     ↓  worker chạy OCR
enriching   (rev 1, OCR xong, có ocr.text + extraction local)
     ↓  worker chạy LLM + matching
completed   (rev 2, full result)
```

File nhỏ (< threshold) vẫn chạy OCR sync như cũ (revision 1 có ngay OCR text trong 202 response). Backward-compat: set `LARGE_FILE_THRESHOLD_BYTES=0` hoặc bỏ env → luôn sync.

Kiểm chứng bằng benchmark 13 file (mỗi file >5MB đi async, 7 file nhỏ vẫn sync): p95 e2e giảm từ **16.1s → 11.3s (-30%)**, submit time file lớn từ 5–13s → ~30ms (100× nhanh hơn).

## Benchmark p95 cho V2

Script đo p50/p95/p99 end-to-end qua API (dùng `debug_timing=true` để lấy stage timings):

```bash
# Cần API + worker đang chạy; --dir trỏ tới thư mục chứa PDF (>= 30 file đa dạng để đo p95 ổn định)
python scripts/benchmark_v2.py \
  --dir data/benchmark \
  --base-url http://localhost:8000 \
  --project agribank --type document \
  --concurrency 8 --iterations 3 \
  --csv /tmp/benchmark_v2.csv
```

Output gồm phân phối `ocr_provider` (cloud_vision / document_ai_fallback / document_ai), p50/p95/p99 end-to-end và từng stage trong `timings.stages`. Acceptance: `p95_ms < 5000` với `OCR_PROVIDER_MODE=vision_primary` và tỉ lệ `document_ai_fallback < 5%`.

### So sánh thuần OCR API (Vision vs Document AI)

Bỏ qua API server và enrichment, gọi trực tiếp 2 backend OCR để đo per-file latency:

```bash
python scripts/benchmark_ocr_apis.py --dir data/benchmark --runs 1
```

Báo cáo bảng per-file (`vision ms` vs `docai ms`, file nào nhanh hơn mấy lần) + summary p50/p95/max. Trên dataset 13 file thực (0.2MB–12MB), Vision nhanh hơn Document AI 1.9–6.5× cho mọi tier.

### So sánh thuật toán split PDF

Khi cần thay backend split (pypdf hiện tại), benchmark 3 library trên dataset thực:

```bash
python scripts/benchmark_pdf_split.py --dir data/benchmark --max-pages 5 --repeats 3
```

Kết quả tham khảo: pikepdf nhanh hơn pypdf ~1.6×, PyMuPDF (AGPL) ~1.1×. Tuy nhiên split chỉ chiếm <0.1% wall time so với OCR API call — không đáng đổi trừ khi PDF cực lớn (50+ trang).

