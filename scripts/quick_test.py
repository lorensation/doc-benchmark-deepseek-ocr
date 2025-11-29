#!/usr/bin/env python3
"""Quick test script for OCR services"""
import requests
import sys
import time
from pathlib import Path

def test_ocr(image_path, service_url, service_name):
    """Test OCR service with a single image"""
    print(f"\n{'='*60}")
    print(f"Testing {service_name}")
    print(f"{'='*60}")
    
    try:
        with open(image_path, 'rb') as f:
            start = time.time()
            response = requests.post(
                service_url,
                files={'file': (Path(image_path).name, f)},
                timeout=60
            )
            elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Status: Success")
        print(f"✓ Response time: {elapsed:.2f}s")
        print(f"✓ Text length: {len(result.get('text', ''))} characters")
        print(f"✓ Confidence: {result.get('confidence', 'N/A')}")
        print(f"\nExtracted text preview (first 200 chars):")
        print("-" * 60)
        print(result.get('text', '')[:200])
        print("-" * 60)
        
        return True, elapsed, result
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, 0, None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/quick_test.py <image_path>")
        print("\nExample:")
        print("  python scripts/quick_test.py data/datasets/invoices_raw/batch1-0001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\nTesting image: {image_path}")
    
    # Test both services
    deepseek_ok, ds_time, ds_result = test_ocr(
        image_path, 
        "http://localhost:9000/ocr",
        "DeepSeek-OCR"
    )
    
    tesseract_ok, ts_time, ts_result = test_ocr(
        image_path,
        "http://localhost:9001/ocr",
        "Tesseract"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"DeepSeek:  {'✓' if deepseek_ok else '✗'}  ({ds_time:.2f}s)")
    print(f"Tesseract: {'✓' if tesseract_ok else '✗'}  ({ts_time:.2f}s)")
    
    if deepseek_ok and tesseract_ok:
        ds_len = len(ds_result.get('text', ''))
        ts_len = len(ts_result.get('text', ''))
        print(f"\nText length comparison:")
        print(f"  DeepSeek:  {ds_len} chars")
        print(f"  Tesseract: {ts_len} chars")
        print(f"  Difference: {abs(ds_len - ts_len)} chars ({abs(ds_len - ts_len) / max(ds_len, ts_len) * 100:.1f}%)")
