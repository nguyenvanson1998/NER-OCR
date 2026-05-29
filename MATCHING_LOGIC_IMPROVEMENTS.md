# 🎯 Matching Logic Improvements - Chi tiết các thay đổi

## 📋 Tổng quan vấn đề

### Vấn đề ban đầu
Document với title: 
```
QUYẾT ĐỊNH Phê duyệt thiết kế xây dựng triển khai sau thiết kế cơ sở và dự toán gói thầu Công trình
```

**Match SAI** với task:
```
❌ Thẩm định và phê duyệt quyết toán dự án hoàn thành
```

Thay vì match ĐÚNG với task:
```
✅ Quyết định phê duyệt thiết kế xây dựng triển khai sau thiết kế cơ sở và dự toán gói thầu
```

### Nguyên nhân
1. **Logic ưu tiên substring ngắn** thay vì matching nhiều keywords
2. **STOPWORDS quá aggressive** - loại bỏ các domain keywords quan trọng
3. **Không đủ bonus cho chuỗi dài** và nhiều keywords matched

---

## 🔧 Các thay đổi chính

### 1. Sửa STOPWORDS (file: `app/services/agribank_matching.py`)

#### Trước khi sửa (61 stopwords):
Loại bỏ các từ quan trọng:
- ❌ `"phe"` → mất "Phê" (từ "Phê duyệt" = approve)
- ❌ `"quyet"` → mất "Quyết" (từ "QUYẾT ĐỊNH" = decision document)
- ❌ `"thau"` → mất "Thầu" (từ "gói thầu" = bidding package)
- ❌ `"trinh"` → mất "Trình" (từ "tờ trình" = proposal)
- ❌ `"goi"` → mất "Gói" (từ "gói thầu" = package)
- ❌ `"cong"` → mất "Công" (từ "công trình" = construction)
- ❌ `"bao"` → mất "Báo" (từ "báo cáo" = report)
- ❌ `"ban"` → mất "Ban" (từ "ban quản lý" = management board)
- ❌ `"hang"` → mất "Hạng" (từ "hạng mục" = work item)
- ❌ `"muc"` → mất "Mục" (từ "hạng mục" = work item)
- ❌ `"thong"` → mất "Thông" (từ "thông báo" = notification)

**Kết quả:** Document title có 20 tokens → chỉ còn 10 tokens sau khi filter!

#### Sau khi sửa (48 stopwords):
Chỉ giữ lại **stopwords thực sự generic**:
```python
STOPWORDS = {
    "a", "an", "cac", "can", "cho", "chi", "cp", "cua", "da", "de", 
    "den", "du", "duoc", "gia", "gui", "ho", "huyen", "ke", "kem", 
    "la", "lap", "nam", "ngay", "ngan", "nghiep", "nhanh", "nong", 
    "noi", "qua", "so", "tai", "theo", "tinh", "tnhh", "thuoc", 
    "to", "trong", "tru", "tu", "ty", "mtv", "va", "ve", "viec", 
    "viet", "xa", "agribank"
}
```

**Kết quả:** Document title có 20 tokens → còn 16 tokens (giữ được 80% semantic keywords!)

---

### 2. Enhanced Matching Logic

#### A. Thêm hàm `longest_common_token_sequence()`
Tìm chuỗi tokens liên tiếp chung dài nhất:
```python
def longest_common_token_sequence(tokens1: list[str], tokens2: list[str]) -> int:
    """Find longest consecutive matching token sequence"""
```

**Ví dụ:**
- Query: `["quyet", "dinh", "phe", "duyet", "thiet", "ke"]`
- Candidate A: `["tham", "dinh", "phe", "duyet", "toan"]` → sequence = 3 (`dinh phe duyet`)
- Candidate B: `["quyet", "dinh", "phe", "duyet", "thiet", "ke"]` → sequence = 6 (full match)

#### B. Longest Sequence Bonus
```python
if longest_seq >= 6:  longest_seq_bonus = 0.12
elif longest_seq >= 4: longest_seq_bonus = 0.08
elif longest_seq >= 3: longest_seq_bonus = 0.04
```

#### C. Matched Token Count Bonus
Thưởng điểm cho matching nhiều keywords:
```python
if matched_token_count >= 8: matched_count_bonus = 0.10
elif matched_token_count >= 6: matched_count_bonus = 0.06
elif matched_token_count >= 4: matched_count_bonus = 0.03
```

#### D. Enhanced Exact Score với Length Bonus
```python
# Trước:
if candidate_norm in query_norm: exact_score = 0.96  # Cố định

# Sau:
if candidate_norm in query_norm:
    base_exact = 0.92
    length_bonus = min(0.08, (matched_token_count / len(query_tokens)) * 0.10)
    exact_score = base_exact + length_bonus  # Tăng theo số tokens matched
```

#### E. Tăng Weight cho Token Score
```python
# Trước:
token_score = 0.50 * candidate_coverage + 0.35 * query_coverage + 0.15 * jaccard

# Sau (ưu tiên query_coverage hơn):
token_score = 0.40 * query_coverage + 0.35 * candidate_coverage + 0.25 * jaccard
```

#### F. Điều chỉnh Weighted Score
```python
# Trước:
weighted = 0.32 * rapid + 0.32 * tfidf + 0.28 * token + 0.08 * sequence

# Sau (tăng weight cho token matching):
weighted = 0.28 * rapid + 0.30 * tfidf + 0.35 * token + 0.07 * sequence
```

---

## 📊 Kết quả So sánh

### Test Case: Document Matching với 5 Tasks

**Document:** "QUYẾT ĐỊNH Phê duyệt thiết kế xây dựng triển khai sau thiết kế cơ sở và dự toán gói thầu Công trình"

| Rank | Task ID | Task Name | Score | Matched Tokens | Query Coverage | Longest Seq | Status |
|------|---------|-----------|-------|----------------|----------------|-------------|---------|
| 🏆 1 | **#3** | **Quyết định phê duyệt thiết kế xây dựng...** | **1.0000** | **14/16** | **87.5%** | **15** | ✅ CORRECT |
| 2 | #1 | Thẩm định và phê duyệt quyết toán... | 0.5228 | 5/16 | 31.25% | 4 | ❌ Wrong |
| 3 | #4 | Lập và phê duyệt dự án đầu tư... | 0.5190 | 6/16 | 37.50% | 2 | ❌ Wrong |
| 4 | #5 | Phê duyệt kết quả lựa chọn nhà thầu | 0.3574 | 3/16 | 18.75% | 2 | ❌ Wrong |
| 5 | #2 | Thực hiện khảo sát xây dựng | 0.3243 | 2/16 | 12.50% | 2 | ❌ Wrong |

### Score Breakdown - Task #3 (CORRECT)
```
Overall Score: 1.0000
├─ exact:                 0.9950  ← Base exact + length bonus
├─ longest_seq_bonus:     0.1200  ← Sequence of 15 tokens
├─ matched_count_bonus:   0.1000  ← 14 matched tokens
├─ token_score:           0.8375  ← High query/candidate coverage
├─ rapidfuzz:             0.9500
└─ tfidf:                 0.0000
```

### Score Breakdown - Task #1 (WRONG)
```
Overall Score: 0.5228
├─ exact:                 0.0000  ← No exact match
├─ longest_seq_bonus:     0.0800  ← Only 4 tokens sequence
├─ matched_count_bonus:   0.0300  ← Only 5 matched tokens
├─ token_score:           0.4095  ← Low coverage
├─ rapidfuzz:             0.8550
└─ tfidf:                 0.0000
```

---

## ✅ Kết luận

### Cải thiện đạt được:
1. ✅ **Ưu tiên matching nhiều keywords** thay vì substring ngắn
2. ✅ **Giữ lại domain keywords** quan trọng (phê, quyết, thầu, trình, gói, công, báo)
3. ✅ **Thưởng điểm cho chuỗi dài** và nhiều tokens matched
4. ✅ **Match chính xác** task đúng với score 1.0000 vs các task sai 0.3-0.5

### Score Gap:
- Trước: Task đúng và sai có thể có score gần nhau
- Sau: **Task đúng 1.0000 vs Task sai cao nhất 0.5228** → gap rõ ràng +0.4772

### Files đã thay đổi:
- ✅ `app/services/agribank_matching.py`:
  - Sửa STOPWORDS (line 73-127)
  - Thêm `longest_common_token_sequence()` (line 880-902)
  - Sửa `best_hybrid_score()` (line 887-921) - ưu tiên queries dài hơn
  - Sửa `hybrid_score()` (line 924-1088) - thêm short query penalty
  - Enhanced `rank_tasks()` (line 665-728) - log top 10 matches

---

## 🆕 **UPDATE: Sửa vấn đề Short Query Penalty**

### Vấn đề phát hiện thêm:
**Queries ngắn từ entities** (như "khảo sát xây dựng") match với task sai do:
1. ❌ `query_coverage = 1.0` (100% query matched)
2. ❌ `exact_score = 0.92+` (substring match)
3. ❌ Score cuối = 1.0000 (giống task đúng!)

**Ví dụ thực tế:**
```json
"task_title_candidates": [
  "Phê duyệt thiết kế xây dựng triển khai sau thiết kế cơ sở và dự toán gói thầu",  ✅ Long
  "khảo sát xây dựng"  ❌ Short
]
```

Query ngắn "khảo sát xây dựng" (4 tokens) match perfect với task "Thực hiện khảo sát xây dựng" → score 1.0

### Giải pháp: Short Query Penalty

#### 1. Aggressive Penalty cho queries ngắn
```python
if query_token_count < 6:
    # 5 tokens: -0.05, 4 tokens: -0.10, 3 tokens: -0.15, 2 tokens: -0.20, 1 token: -0.25
    short_query_penalty = (6 - query_token_count) * 0.05
```

#### 2. Prefer Longer Queries khi scores gần nhau
```python
def best_hybrid_score(queries, candidate, tfidf_scores):
    # When scores differ by < 0.02, prefer longer query
    if score_diff > -0.02:
        if current_query_tokens > best_query_tokens:
            best_payload = current_payload  # Longer query wins
```

### Kết quả sau khi sửa:

| Task Name | Matched Query | Score TRƯỚC | Score SAU | Status |
|-----------|---------------|-------------|-----------|--------|
| **Quyết định phê duyệt thiết kế...** | QUYẾT ĐỊNH Phê duyệt... (16 tokens) | 1.0000 | **1.0000** | ✅ CORRECT |
| Thực hiện khảo sát xây dựng | khảo sát xây dựng (4 tokens) | **1.0000** ❌ | **0.9318** ✓ | Dropped to #3 |
| Khảo sát xây dựng | khảo sát xây dựng (4 tokens) | **1.0000** ❌ | **0.9367** ✓ | Dropped to #2 |

**Score Gap:** Task đúng (1.0000) vs task sai cao nhất (0.9367) = **+0.0633**

---

## 📝 **Summary of All Changes**

### A. STOPWORDS (Giữ domain keywords)
- **Removed:** `phe`, `quyet`, `thau`, `trinh`, `goi`, `cong`, `bao`, `ban`, `hang`, `muc`, `thong` (13 keywords)
- **Impact:** 10/20 tokens → **16/20 tokens** retained (80%)

### B. Matching Logic Enhancements
1. ✅ `longest_common_token_sequence()` - find consecutive matching tokens
2. ✅ **Longest sequence bonus** (+0.04 to +0.12)
3. ✅ **Matched count bonus** (+0.03 to +0.10)
4. ✅ **Enhanced exact score** with length bonus
5. ✅ **Increased token_score weight** (0.28 → 0.35)
6. ✅ **Increased query_coverage weight** (0.35 → 0.40)
7. 🆕 **Short query penalty** (-0.05 to -0.25 for queries < 6 tokens)
8. 🆕 **Prefer longer queries** when scores differ by < 0.02

### C. Task Query Strategy Change
9. 🆕 **Use ONLY document title** for task matching (not LLM entities)

**Previous behavior:**
```python
task_queries = [
  "QUYẾT ĐỊNH Phê duyệt thiết kế...",  # From title
  "khảo sát xây dựng",  # From entities.work_items ❌
  "gói thầu thi công",  # From entities.procurement_packages ❌
]
```

**New behavior:**
```python
task_queries = [
  "QUYẾT ĐỊNH Phê duyệt thiết kế xây dựng...",  # Title (full)
  "QUYẾT ĐỊNH Phê duyệt thiết kế xây dựng...",  # Title without "về việc"
  "QUYẾT ĐỊNH Phê duyệt thiết kế...",  # Title without "công trình: ..."
]
```

**Rationale:**
- LLM entities (`work_items`, `business_actions`) are often too generic
- Short queries get `query_coverage=1.0` and match wrong tasks
- Document title is the single source of truth

### D. Debugging Tools
- ✅ **Top 10 task logging** in `rank_tasks()` with full breakdown
