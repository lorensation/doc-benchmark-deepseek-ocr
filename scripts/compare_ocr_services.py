#!/usr/bin/env python3
"""Compare OCR services side-by-side on a single image"""
import requests
import sys
import time
from pathlib import Path
import json
from datetime import datetime

def call_ocr_service(image_path, service_url, service_name):
    """Call OCR service and return results"""
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
        
        return {
            'success': True,
            'service': service_name,
            'response_time': elapsed,
            'text': result.get('text', ''),
            'confidence': result.get('confidence', None),
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'service': service_name,
            'response_time': 0,
            'text': '',
            'confidence': None,
            'error': str(e)
        }

def print_diff(text1, text2, name1, name2):
    """Print simple character-level difference"""
    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    
    print(f"\n{'='*60}")
    print("LINE-BY-LINE COMPARISON (first 10 lines)")
    print(f"{'='*60}")
    
    max_lines = min(len(lines1), len(lines2), 10)
    
    for i in range(max_lines):
        if i < len(lines1) and i < len(lines2):
            if lines1[i] == lines2[i]:
                print(f"Line {i+1}: ✓ Match")
            else:
                print(f"Line {i+1}: ✗ Different")
                print(f"  {name1}: {lines1[i][:60]}")
                print(f"  {name2}: {lines2[i][:60]}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_ocr_services.py <image_path>")
        print("\nExample:")
        print("  python scripts/compare_ocr_services.py data/datasets/invoices_raw/batch1-0001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"OCR SERVICE COMPARISON")
    print(f"{'='*60}")
    print(f"Image: {Path(image_path).name}")
    
    # Call both services
    print("\nCalling DeepSeek-OCR...")
    deepseek_result = call_ocr_service(image_path, "http://localhost:9000/ocr", "DeepSeek-OCR")
    
    print("Calling Tesseract...")
    tesseract_result = call_ocr_service(image_path, "http://localhost:9001/ocr", "Tesseract")
    
    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    for result in [deepseek_result, tesseract_result]:
        print(f"\n{result['service']}:")
        if result['success']:
            print(f"  ✓ Success")
            print(f"  Response time: {result['response_time']:.2f}s")
            print(f"  Text length: {len(result['text'])} characters")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Text preview (first 200 chars):")
            print(f"  {'-'*58}")
            print(f"  {result['text'][:200]}")
            print(f"  {'-'*58}")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Comparison metrics
    if deepseek_result['success'] and tesseract_result['success']:
        print(f"\n{'='*60}")
        print("COMPARISON METRICS")
        print(f"{'='*60}")
        
        ds_len = len(deepseek_result['text'])
        ts_len = len(tesseract_result['text'])
        
        print(f"\nText Length:")
        print(f"  DeepSeek:  {ds_len} chars")
        print(f"  Tesseract: {ts_len} chars")
        print(f"  Difference: {abs(ds_len - ts_len)} chars ({abs(ds_len - ts_len) / max(ds_len, ts_len) * 100:.1f}%)")
        
        print(f"\nResponse Time:")
        print(f"  DeepSeek:  {deepseek_result['response_time']:.2f}s")
        print(f"  Tesseract: {tesseract_result['response_time']:.2f}s")
        speed_ratio = deepseek_result['response_time'] / tesseract_result['response_time'] if tesseract_result['response_time'] > 0 else 0
        print(f"  Ratio: {speed_ratio:.2f}x (DeepSeek / Tesseract)")
        
        # Simple diff
        print_diff(deepseek_result['text'], tesseract_result['text'], "DeepSeek", "Tesseract")
        
        # Save comparison (use absolute path from script location)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        results_dir = project_root / "data" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"comparison_{Path(image_path).stem}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'image': str(image_path),
                'deepseek': deepseek_result,
                'tesseract': tesseract_result
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Detailed comparison saved to: {output_file}")
