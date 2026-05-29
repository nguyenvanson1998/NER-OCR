"""
Comprehensive test script for Google Cloud credentials.
Tests: Authentication, Document AI, Gemini API, and Vertex AI Embeddings.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

# Set credentials path
creds_path = BASE_DIR / 'credentials' / 'credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_path)

# Load project configuration
PROJECT_ID = os.getenv('GOOGLE_AI_PROJECT_ID', 'peppy-sensor-267116')
LOCATION = os.getenv('GOOGLE_AI_LOCATION', 'us')
VERTEX_LOCATION = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
PROCESSOR_ID = os.getenv('GOOGLE_AI_PROCESSOR_ID', '67eca44d0a36dc87')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME', 'gemini-embedding-001')

print("=" * 80)
print("🧪 GOOGLE CLOUD CREDENTIALS COMPREHENSIVE TEST")
print("=" * 80)
print()

# Load credentials info
with open(creds_path) as f:
    creds = json.load(f)

print('📋 Thông tin credentials:')
print(f'  - Project ID: {creds.get("project_id")}')
print(f'  - Client Email: {creds.get("client_email")}')
print(f'  - Private Key ID: {creds.get("private_key_id")}')
print(f'  - Service Account Type: {creds.get("type")}')
print()

print('📋 Cấu hình từ .env:')
print(f'  - Project ID: {PROJECT_ID}')
print(f'  - Document AI Location: {LOCATION}')
print(f'  - Vertex AI Location: {VERTEX_LOCATION}')
print(f'  - Processor ID: {PROCESSOR_ID}')
print(f'  - Embedding Model: {EMBEDDING_MODEL}')
print()

print("=" * 80)


# ============================================================================
# TEST 1: Basic Authentication
# ============================================================================
def test_basic_auth():
    """Test basic OAuth2 authentication with Google Cloud."""
    print("\n🔐 TEST 1: BASIC AUTHENTICATION")
    print("-" * 80)

    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        credentials = service_account.Credentials.from_service_account_file(
            str(creds_path),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        # Refresh to get token
        credentials.refresh(Request())

        print('✅ PASSED: Credentials hợp lệ!')
        print(f'   - Token được tạo thành công')
        print(f'   - Token expiry: {credentials.expiry}')
        return True

    except Exception as e:
        print(f'❌ FAILED: Credentials KHÔNG hợp lệ!')
        print(f'   - Lỗi: {type(e).__name__}')
        print(f'   - Chi tiết: {str(e)}')
        return False


# ============================================================================
# TEST 2: Document AI API
# ============================================================================
def test_document_ai():
    """Test Document AI processor access."""
    print("\n📄 TEST 2: DOCUMENT AI API")
    print("-" * 80)

    try:
        from google.cloud import documentai_v1 as documentai
        from google.api_core import exceptions

        client = documentai.DocumentProcessorServiceClient()
        processor_name = f'projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}'

        print(f'🔍 Đang kiểm tra processor: {processor_name}')

        try:
            processor = client.get_processor(name=processor_name)
            print(f'✅ PASSED: Document AI accessible!')
            print(f'   - Display Name: {processor.display_name}')
            print(f'   - Type: {processor.type_}')
            print(f'   - State: {processor.state.name}')
            return True

        except exceptions.PermissionDenied:
            print(f'❌ FAILED: KHÔNG có quyền truy cập Document AI')
            print(f'   - Service account cần được cấp quyền "Document AI User" hoặc "Document AI Editor"')
            return False

        except exceptions.NotFound:
            print(f'❌ FAILED: Processor không tồn tại')
            print(f'   - Kiểm tra lại PROCESSOR_ID trong .env')
            return False

    except ImportError as e:
        print(f'❌ FAILED: Thiếu thư viện google-cloud-documentai')
        print(f'   - Chạy: pip install google-cloud-documentai')
        return False

    except Exception as e:
        print(f'❌ FAILED: Lỗi không xác định')
        print(f'   - Lỗi: {type(e).__name__}')
        print(f'   - Chi tiết: {str(e)}')
        return False


# ============================================================================
# TEST 3: Gemini API (Vertex AI)
# ============================================================================
def test_gemini_api():
    """Test Gemini generative model via Vertex AI."""
    print("\n🤖 TEST 3: GEMINI API (VERTEX AI)")
    print("-" * 80)

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        print(f'🔍 Đang khởi tạo Vertex AI với project: {PROJECT_ID}, location: {VERTEX_LOCATION}')

        try:
            # Initialize Vertex AI
            vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)

            # Try to load a model
            model = GenerativeModel("gemini-2.5-flash")

            print(f'✅ Model loaded: gemini-2.5-flash')

            # Test a simple generation
            print(f'🧪 Testing text generation...')
            response = model.generate_content("Say 'Hello' in Vietnamese")

            print(f'✅ PASSED: Gemini API accessible!')
            print(f'   - Model: gemini-2.5-flash')
            print(f'   - Response: {response.text[:100] if response.text else "No text response"}')
            return True

        except Exception as e:
            error_msg = str(e).lower()
            error_full = str(e)

            if '401' in error_msg or 'unauthenticated' in error_msg:
                print(f'❌ FAILED: Authentication error')
                print(f'   - Credentials không hợp lệ hoặc thiếu quyền')

            elif '403' in error_msg or 'permission denied' in error_msg:
                print(f'❌ FAILED: Permission denied')
                print(f'   - Service account cần quyền "Vertex AI User"')

            elif '404' in error_msg or 'not found' in error_msg:
                print(f'❌ FAILED: Model không được phép sử dụng')
                print(f'   - Project này không có quyền truy cập Gemini models')
                print(f'   - Cần enable Vertex AI API và đăng ký sử dụng Gemini')
                print(f'   - Hoặc cấu hình GEMINI_API_KEY nếu dùng Gemini API key trực tiếp')

            else:
                print(f'❌ FAILED: {type(e).__name__}')
                print(f'   - Chi tiết: {error_full}')

            return False

    except ImportError:
        print(f'❌ FAILED: Thiếu thư viện google-cloud-aiplatform')
        print(f'   - Chạy: pip install google-cloud-aiplatform')
        return False


# ============================================================================
# TEST 4: Vertex AI Embeddings
# ============================================================================
def test_vertex_embeddings():
    """Test Vertex AI text embedding generation."""
    print("\n🔢 TEST 4: VERTEX AI EMBEDDINGS")
    print("-" * 80)

    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        print(f'🔍 Đang khởi tạo Vertex AI Embeddings...')
        print(f'   - Model: {EMBEDDING_MODEL}')

        try:
            # Initialize Vertex AI
            vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)

            # Load embedding model
            model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

            # Test embedding generation
            test_text = "This is a test sentence for embedding."
            print(f'🧪 Testing embedding generation...')

            embeddings = model.get_embeddings([test_text])

            print(f'✅ PASSED: Vertex AI Embeddings accessible!')
            print(f'   - Model: {EMBEDDING_MODEL}')
            print(f'   - Embedding dimension: {len(embeddings[0].values)}')
            print(f'   - First 5 values: {embeddings[0].values[:5]}')
            return True

        except Exception as e:
            error_msg = str(e).lower()

            if '401' in error_msg or 'unauthenticated' in error_msg:
                print(f'❌ FAILED: Authentication error')
                print(f'   - Credentials không hợp lệ')

            elif '403' in error_msg or 'permission denied' in error_msg:
                print(f'❌ FAILED: Permission denied')
                print(f'   - Service account cần quyền "Vertex AI User"')

            elif '404' in error_msg or 'not found' in error_msg:
                print(f'❌ FAILED: Model không tìm thấy')
                print(f'   - Kiểm tra EMBEDDING_MODEL_NAME trong .env')

            else:
                print(f'❌ FAILED: {type(e).__name__}')
                print(f'   - Chi tiết: {str(e)}')

            return False

    except ImportError:
        print(f'❌ FAILED: Thiếu thư viện google-cloud-aiplatform')
        print(f'   - Chạy: pip install google-cloud-aiplatform')
        return False


# ============================================================================
# RUN ALL TESTS
# ============================================================================
if __name__ == "__main__":
    results = {
        "Basic Auth": test_basic_auth(),
        "Document AI": test_document_ai(),
        "Gemini API": test_gemini_api(),
        "Vertex Embeddings": test_vertex_embeddings(),
    }

    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Credentials are fully functional.")
    elif passed > 0:
        print("⚠️  Some tests failed. Check permissions and API enablement.")
    else:
        print("❌ All tests failed. Credentials are invalid or lack permissions.")

    print("=" * 80)
