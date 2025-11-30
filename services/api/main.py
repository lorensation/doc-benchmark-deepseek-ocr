from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
import uuid
import shutil
import json
import datetime as dt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import mimetypes
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "api"}'
)
logger = logging.getLogger(__name__)


# Configuration with validation
class Settings(BaseSettings):
    upload_dir: str = Field(default="/app/data/uploads")
    results_dir: str = Field(default="/app/data/results")
    deepseek_url: str = Field(default="http://deepseek:9000/ocr")
    tesseract_url: str = Field(default="http://tesseract:9001/ocr")
    vista_url: str = Field(default="http://vista:9002/ocr")
    hunyuan_url: str = Field(default="http://hunyuan:9003/ocr")
    qwen_url: str = Field(default="http://qwen2vl:9004/ocr")
    max_file_size_mb: int = Field(default=10)
    request_timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    allowed_extensions: List[str] = Field(default=["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"])
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.results_dir, exist_ok=True)

# Configure requests session with retry logic
def create_retry_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=settings.max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


retry_session = create_retry_session()

app = FastAPI(
    title="Doc Benchmark API",
    version="0.2.0",
    description="Production-grade OCR benchmarking API with comprehensive error handling"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP exception: {exc.detail}", extra={"status_code": exc.status_code})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


class BenchmarkResult(BaseModel):
    run_id: str
    filename: str
    deepseek_text: str
    tesseract_text: str
    vista_text: str = ""
    hunyuan_text: str = ""
    qwen2vl_text: str = ""
    length_deepseek: int
    length_tesseract: int
    length_vista: int = 0
    length_hunyuan: int = 0
    length_qwen2vl: int = 0
    created_at: str
    status: str = Field(default="completed")
    error: Optional[str] = None


class RunListItem(BaseModel):
    run_id: str
    filename: str
    created_at: str
    status: str = Field(default="completed")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file for security and format compliance."""
    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Sanitize filename
    safe_filename = re.sub(r'[^\w\s\-\.]', '', file.filename)
    if not safe_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower().lstrip('.')
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type .{file_ext} not allowed. Allowed types: {', '.join(settings.allowed_extensions)}"
        )
    
    # Validate MIME type
    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type and not mime_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type: {mime_type}. Only image files are allowed."
        )


def call_ocr_service(service_url: str, service_name: str, file_path: str, filename: str) -> str:
    """Call OCR service with error handling and retry logic."""
    try:
        logger.info(f"Calling {service_name} service for {filename}")
        
        with open(file_path, "rb") as f:
            response = retry_session.post(
                service_url,
                files={"file": (filename, f)},
                timeout=settings.request_timeout
            )
        
        response.raise_for_status()
        result = response.json()
        text = result.get("text", "")
        
        logger.info(f"{service_name} completed for {filename}: {len(text)} characters extracted")
        return text
        
    except requests.exceptions.Timeout:
        logger.error(f"{service_name} timeout for {filename}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"{service_name} service timeout after {settings.request_timeout}s"
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"{service_name} connection error for {filename}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service_name} service unavailable"
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"{service_name} HTTP error for {filename}: {e}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{service_name} service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"{service_name} unexpected error for {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{service_name} service error: {str(e)}"
        )


@app.post("/upload_and_benchmark", response_model=BenchmarkResult)
async def upload_and_benchmark(file: UploadFile = File(...)):
    """Upload an image and run OCR benchmark across all configured OCR services."""
    run_id = str(uuid.uuid4())
    correlation_id = run_id
    dest_path = None
    
    try:
        logger.info(f"Starting benchmark {correlation_id} for file: {file.filename}")
        
        # Validate file
        validate_file(file)
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed size ({settings.max_file_size_mb}MB)"
            )
        
        logger.info(f"File validation passed for {file.filename}: {file_size / 1024:.2f}KB")
        
        # Save file with sanitized name
        safe_filename = re.sub(r'[^\w\s\-\.]', '', file.filename)
        filename = f"{run_id}_{safe_filename}"
        dest_path = os.path.join(settings.upload_dir, filename)
        
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        
        logger.info(f"File saved to {dest_path}")
        
        # Call OCR services
        ocr_services = [
            ("DeepSeek", settings.deepseek_url, "deepseek_text", "length_deepseek"),
            ("Tesseract", settings.tesseract_url, "tesseract_text", "length_tesseract"),
            ("VISTA-OCR", settings.vista_url, "vista_text", "length_vista"),
            ("HunyuanOCR", settings.hunyuan_url, "hunyuan_text", "length_hunyuan"),
            ("Qwen2-VL", settings.qwen_url, "qwen2vl_text", "length_qwen2vl"),
        ]
        ocr_results = {}
        for service_name, url, text_key, length_key in ocr_services:
            text = call_ocr_service(url, service_name, dest_path, filename)
            ocr_results[text_key] = text
            ocr_results[length_key] = len(text)
        
        created_at = dt.datetime.utcnow().isoformat()
        
        result = {
            "run_id": run_id,
            "filename": filename,
            **ocr_results,
            "created_at": created_at,
            "status": "completed"
        }
        
        # Save result
        out_path = os.path.join(settings.results_dir, f"{run_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Benchmark {correlation_id} completed successfully")
        return BenchmarkResult(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Benchmark {correlation_id} failed: {str(e)}", exc_info=True)
        
        # Save failed result
        failed_result = {
            "run_id": run_id,
            "filename": file.filename,
            "deepseek_text": "",
            "tesseract_text": "",
            "vista_text": "",
            "hunyuan_text": "",
            "qwen2vl_text": "",
            "length_deepseek": 0,
            "length_tesseract": 0,
            "length_vista": 0,
            "length_hunyuan": 0,
            "length_qwen2vl": 0,
            "created_at": dt.datetime.utcnow().isoformat(),
            "status": "failed",
            "error": str(e)
        }
        
        try:
            out_path = os.path.join(settings.results_dir, f"{run_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(failed_result, f, ensure_ascii=False, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save error result: {save_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {str(e)}"
        )
    finally:
        # Cleanup on error (optional - keep file for debugging)
        pass


@app.get("/runs", response_model=List[RunListItem])
def list_runs():
    """List all benchmark runs with pagination support."""
    try:
        logger.info("Fetching benchmark runs list")
        items: List[RunListItem] = []
        
        if not os.path.exists(settings.results_dir):
            logger.warning(f"Results directory does not exist: {settings.results_dir}")
            return items
        
        for fn in os.listdir(settings.results_dir):
            if not fn.endswith(".json"):
                continue
            
            try:
                file_path = os.path.join(settings.results_dir, fn)
                with open(file_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                
                items.append(
                    RunListItem(
                        run_id=d.get("run_id", ""),
                        filename=d.get("filename", ""),
                        created_at=d.get("created_at", ""),
                        status=d.get("status", "completed")
                    )
                )
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in file {fn}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error reading file {fn}: {e}")
                continue
        
        # Sort by date desc
        items.sort(key=lambda x: x.created_at, reverse=True)
        logger.info(f"Found {len(items)} benchmark runs")
        return items
        
    except Exception as e:
        logger.error(f"Error listing runs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving runs: {str(e)}"
        )


@app.get("/runs/{run_id}", response_model=BenchmarkResult)
def get_run(run_id: str):
    """Get details of a specific benchmark run."""
    try:
        # Validate run_id format (UUID)
        try:
            uuid.UUID(run_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid run_id format. Must be a valid UUID."
            )
        
        logger.info(f"Fetching benchmark run: {run_id}")
        path = os.path.join(settings.results_dir, f"{run_id}.json")
        
        if not os.path.exists(path):
            logger.warning(f"Benchmark run not found: {run_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark run {run_id} not found"
            )
        
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        
        logger.info(f"Successfully retrieved benchmark run: {run_id}")
        return BenchmarkResult(**d)
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in run {run_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Corrupted benchmark data"
        )
    except Exception as e:
        logger.error(f"Error retrieving run {run_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving run: {str(e)}"
        )


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "api",
        "version": "0.2.0",
        "timestamp": dt.datetime.utcnow().isoformat()
    }


@app.get("/ready")
def readiness_check():
    """Readiness check - verifies dependencies are accessible."""
    checks = {
        "api": "healthy",
        "deepseek": "unknown",
        "tesseract": "unknown",
        "vista": "unknown",
        "hunyuan": "unknown",
        "qwen2vl": "unknown",
        "storage": "unknown"
    }
    
    # Check DeepSeek service
    try:
        resp = requests.get(f"{settings.deepseek_url.replace('/ocr', '/health')}", timeout=5)
        checks["deepseek"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        checks["deepseek"] = "unhealthy"
    
    # Check Tesseract service
    try:
        resp = requests.get(f"{settings.tesseract_url.replace('/ocr', '/health')}", timeout=5)
        checks["tesseract"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        checks["tesseract"] = "unhealthy"

    # Check VISTA service
    try:
        resp = requests.get(f"{settings.vista_url.replace('/ocr', '/health')}", timeout=5)
        checks["vista"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        checks["vista"] = "unhealthy"

    # Check Hunyuan service
    try:
        resp = requests.get(f"{settings.hunyuan_url.replace('/ocr', '/health')}", timeout=5)
        checks["hunyuan"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        checks["hunyuan"] = "unhealthy"

    # Check Qwen2-VL service
    try:
        resp = requests.get(f"{settings.qwen_url.replace('/ocr', '/health')}", timeout=5)
        checks["qwen2vl"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        checks["qwen2vl"] = "unhealthy"
    
    # Check storage
    checks["storage"] = "healthy" if os.path.exists(settings.results_dir) and os.access(settings.results_dir, os.W_OK) else "unhealthy"
    
    all_healthy = all(v == "healthy" for k, v in checks.items() if k != "api")
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "timestamp": dt.datetime.utcnow().isoformat()
        }
    )
