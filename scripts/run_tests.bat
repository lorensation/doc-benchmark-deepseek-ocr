@echo off
REM Quick test runner for Windows
REM This batch file makes it easier to run the Python scripts

echo ========================================
echo OCR Service Testing - Quick Start
echo ========================================
echo.

REM Check if services are running
echo [1/4] Checking if services are running...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ERROR: API service not responding at localhost:8000
    echo Please start services with: docker-compose up -d
    exit /b 1
)
echo   ^> API service is running

curl -s http://localhost:9001/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: Tesseract service not responding at localhost:9001
) else (
    echo   ^> Tesseract service is running
)

curl -s http://localhost:9000/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: DeepSeek service not responding at localhost:9000
    echo Note: DeepSeek takes 3-5 minutes to start on first run
) else (
    echo   ^> DeepSeek service is running
)

echo.
echo [2/4] Validating dataset...
python validate_dataset.py C:\Users\sanzp\Desktop\AI\OCR-Benchmark-ENV\doc-benchmark-deepseek-ocr\data\datasets\invoices_raw
if errorlevel 1 (
    echo ERROR: Dataset validation failed
    exit /b 1
)

echo.
echo [3/4] Running quick test on first image...
python quick_test.py C:\Users\sanzp\Desktop\AI\OCR-Benchmark-ENV\doc-benchmark-deepseek-ocr\data\datasets\invoices_raw\batch1-0001.jpg
if errorlevel 1 (
    echo ERROR: Quick test failed
    exit /b 1
)

echo.
echo [4/4] Running batch test on 5 images...
python batch_quick_test.py C:\Users\sanzp\Desktop\AI\OCR-Benchmark-ENV\doc-benchmark-deepseek-ocr\data\datasets\invoices_raw 5

echo.
echo ========================================
echo Testing Complete!
echo ========================================
echo.
echo Results saved to: data/test_results/
echo.
echo Next steps:
echo   - Review results in data/test_results/
echo   - Test more images: python scripts/batch_quick_test.py data/datasets/invoices_raw/ 15
echo   - Compare services: python scripts/compare_ocr_services.py data/datasets/invoices_raw/batch1-0001.jpg
echo.
pause
