#!/usr/bin/env python3
"""Validate invoice dataset - check image integrity and generate statistics"""
from pathlib import Path
import sys
from PIL import Image
import json
from datetime import datetime

def validate_dataset(dataset_dir):
    """Validate all images in dataset directory"""
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Error: Directory not found: {dataset_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("DATASET VALIDATION")
    print(f"{'='*60}")
    print(f"Directory: {dataset_dir}\n")
    
    # Find all image files
    image_files = list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png'))
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} potential image files\n")
    
    valid_images = []
    invalid_images = []
    format_counts = {}
    total_size = 0
    dimensions = []
    
    print("Validating images...")
    for i, img_path in enumerate(image_files, 1):
        if i % 50 == 0 or i == len(image_files):
            print(f"  Progress: {i}/{len(image_files)}")
        
        try:
            with Image.open(img_path) as img:
                # Get image info
                width, height = img.size
                img_format = img.format
                file_size = img_path.stat().st_size
                
                valid_images.append({
                    'filename': img_path.name,
                    'format': img_format,
                    'width': width,
                    'height': height,
                    'size_bytes': file_size
                })
                
                # Track statistics
                format_counts[img_format] = format_counts.get(img_format, 0) + 1
                total_size += file_size
                dimensions.append((width, height))
                
        except Exception as e:
            invalid_images.append({
                'filename': img_path.name,
                'error': str(e)
            })
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total files:     {len(image_files)}")
    print(f"Valid images:    {len(valid_images)} ✓")
    print(f"Invalid images:  {len(invalid_images)} ✗")
    
    if valid_images:
        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        
        # Format distribution
        print("\nImage Formats:")
        for fmt, count in sorted(format_counts.items()):
            print(f"  {fmt}: {count} images ({count/len(valid_images)*100:.1f}%)")
        
        # Size statistics
        sizes = [img['size_bytes'] for img in valid_images]
        print(f"\nFile Sizes:")
        print(f"  Total: {total_size / (1024*1024):.2f} MB")
        print(f"  Min: {min(sizes) / 1024:.2f} KB")
        print(f"  Max: {max(sizes) / 1024:.2f} KB")
        print(f"  Avg: {sum(sizes) / len(sizes) / 1024:.2f} KB")
        
        # Dimension statistics
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        print(f"\nImage Dimensions:")
        print(f"  Width:  Min={min(widths)}, Max={max(widths)}, Avg={sum(widths)//len(widths)}")
        print(f"  Height: Min={min(heights)}, Max={max(heights)}, Avg={sum(heights)//len(heights)}")
        
        # Most common dimensions
        from collections import Counter
        dim_counter = Counter(dimensions)
        print(f"\nMost common dimensions:")
        for dims, count in dim_counter.most_common(5):
            print(f"  {dims[0]}x{dims[1]}: {count} images")
    
    if invalid_images:
        print(f"\n{'='*60}")
        print("INVALID IMAGES")
        print(f"{'='*60}")
        for img in invalid_images:
            print(f"  ✗ {img['filename']}: {img['error']}")
    
    # Save validation report
    results_dir = Path("data/test_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"dataset_validation_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'dataset_directory': str(dataset_dir),
        'total_files': len(image_files),
        'valid_count': len(valid_images),
        'invalid_count': len(invalid_images),
        'format_counts': format_counts,
        'total_size_mb': total_size / (1024*1024),
        'valid_images': valid_images,
        'invalid_images': invalid_images
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Validation report saved to: {output_file}")
    
    return len(valid_images), len(invalid_images)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_dataset.py <dataset_directory>")
        print("\nExample:")
        print("  python scripts/validate_dataset.py data/datasets/invoices_raw/")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    valid_count, invalid_count = validate_dataset(dataset_dir)
    
    if invalid_count > 0:
        sys.exit(1)  # Exit with error if there are invalid images
