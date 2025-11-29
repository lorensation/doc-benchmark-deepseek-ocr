#!/usr/bin/env python3
"""Quick batch test for multiple invoice images"""
import requests
import time
from pathlib import Path
import sys
import json
from datetime import datetime

def batch_test(image_dir, service_url, service_name, limit=5):
    """Test OCR service with multiple images"""
    image_dir = Path(image_dir)
    images = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    images = images[:limit]  # Limit for quick test
    
    print(f"\n{'='*60}")
    print(f"Batch testing {service_name}")
    print(f"Testing {len(images)} images from {image_dir}")
    print(f"{'='*60}\n")
    
    results = []
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Testing {img_path.name}...", end=' ')
        
        try:
            with open(img_path, 'rb') as f:
                start = time.time()
                response = requests.post(
                    service_url,
                    files={'file': (img_path.name, f)},
                    timeout=60
                )
                elapsed = time.time() - start
            
            response.raise_for_status()
            result = response.json()
            text_len = len(result.get('text', ''))
            
            results.append({
                'success': True,
                'time': elapsed,
                'text_length': text_len,
                'filename': img_path.name,
                'confidence': result.get('confidence', None)
            })
            
            print(f"✓ {elapsed:.2f}s ({text_len} chars)")
            
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'filename': img_path.name
            })
            print(f"✗ {str(e)[:50]}")
    
    # Summary statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total images:  {len(results)}")
    print(f"Successful:    {len(successful)} ✓")
    print(f"Failed:        {len(failed)} ✗")
    
    if successful:
        times = [r['time'] for r in successful]
        lengths = [r['text_length'] for r in successful]
        
        print(f"\nResponse Times:")
        print(f"  Min:  {min(times):.2f}s")
        print(f"  Max:  {max(times):.2f}s")
        print(f"  Avg:  {sum(times)/len(times):.2f}s")
        
        print(f"\nText Extraction:")
        print(f"  Min length:  {min(lengths)} chars")
        print(f"  Max length:  {max(lengths)} chars")
        print(f"  Avg length:  {sum(lengths)//len(lengths)} chars")
    
    if failed:
        print(f"\nFailed images:")
        for r in failed:
            print(f"  ✗ {r['filename']}")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/batch_quick_test.py <image_directory> [limit]")
        print("\nExamples:")
        print("  python scripts/batch_quick_test.py data/datasets/invoices_raw/ 5")
        print("  python scripts/batch_quick_test.py data/datasets/invoices_raw/ 15")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test DeepSeek
    print("\n" + "="*60)
    print("TESTING DEEPSEEK-OCR")
    print("="*60)
    deepseek_results = batch_test(
        image_dir,
        "http://localhost:9000/ocr",
        "DeepSeek-OCR",
        limit
    )
    
    # Test Tesseract
    print("\n" + "="*60)
    print("TESTING TESSERACT")
    print("="*60)
    tesseract_results = batch_test(
        image_dir,
        "http://localhost:9001/ocr",
        "Tesseract",
        limit
    )
    
    # Comparison
    ds_success = sum(1 for r in deepseek_results if r['success'])
    ts_success = sum(1 for r in tesseract_results if r['success'])
    
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"DeepSeek success rate:  {ds_success}/{limit} ({ds_success/limit*100:.1f}%)")
    print(f"Tesseract success rate: {ts_success}/{limit} ({ts_success/limit*100:.1f}%)")
    
    # Save results (use absolute path from script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "data" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f"batch_test_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'image_directory': str(image_dir),
            'limit': limit,
            'deepseek_results': deepseek_results,
            'tesseract_results': tesseract_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
