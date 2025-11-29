# OCR Testing Scripts

Quick reference guide for testing OCR services with the invoice dataset.

## Available Scripts

### 1. `validate_dataset.py` - Dataset Validation
Checks image integrity and generates statistics for your dataset.

**Usage:**
```bash
python scripts/validate_dataset.py data/datasets/invoices_raw/
```

**What it does:**
- Validates all images can be opened
- Reports image formats, dimensions, file sizes
- Identifies corrupted files
- Saves validation report to `data/test_results/`

**Run this first** to ensure your dataset is ready for testing!

---

### 2. `quick_test.py` - Single Image Test
Test both OCR services with a single invoice image.

**Usage:**
```bash
python scripts/quick_test.py data/datasets/invoices_raw/batch1-0001.jpg
```

**What it does:**
- Calls both DeepSeek and Tesseract
- Shows response times
- Displays text preview
- Compares text length

**Perfect for:** Quick service verification, testing specific problematic images.

---

### 3. `batch_quick_test.py` - Batch Testing
Test multiple images at once and get performance statistics.

**Usage:**
```bash
# Test 5 images (quick)
python scripts/batch_quick_test.py data/datasets/invoices_raw/ 5

# Test 15 images (recommended for initial testing)
python scripts/batch_quick_test.py data/datasets/invoices_raw/ 15

# Test 50 images (comprehensive)
python scripts/batch_quick_test.py data/datasets/invoices_raw/ 50
```

**What it does:**
- Tests multiple images sequentially
- Calculates min/max/avg response times
- Shows success rates
- Saves detailed results to JSON

**Perfect for:** Performance benchmarking, finding patterns in failures.

---

### 4. `compare_ocr_services.py` - Side-by-Side Comparison
Detailed comparison of both services on a single image.

**Usage:**
```bash
python scripts/compare_ocr_services.py data/datasets/invoices_raw/batch1-0001.jpg
```

**What it does:**
- Calls both services
- Shows full extracted text
- Line-by-line comparison
- Performance metrics
- Saves detailed comparison to JSON

**Perfect for:** Analyzing specific differences, quality evaluation.

---

## Quick Start Workflow

### Step 1: Validate Your Dataset (2 minutes)
```bash
python scripts/validate_dataset.py data/datasets/invoices_raw/
```
This ensures all images are readable and gives you dataset statistics.

### Step 2: Test Single Image (1 minute)
```bash
python scripts/quick_test.py data/datasets/invoices_raw/batch1-0001.jpg
```
Verify both services are working correctly.

### Step 3: Run Batch Test (5-10 minutes)
```bash
python scripts/batch_quick_test.py data/datasets/invoices_raw/ 10
```
Test 10 images to get baseline performance metrics.

### Step 4: Deep Dive (optional)
```bash
python scripts/compare_ocr_services.py data/datasets/invoices_raw/batch1-0001.jpg
```
Analyze specific images in detail.

---

## Prerequisites

### 1. Services Running
Make sure Docker services are running:
```bash
docker-compose ps
```

### 2. Services Healthy
Check service health:
```bash
curl http://localhost:8000/health
curl http://localhost:9000/health
curl http://localhost:9001/health
```

### 3. Python Dependencies
These scripts use only standard library + requests:
```bash
pip install requests Pillow
```

---

## Understanding the Output

### Response Times
- **Tesseract**: Typically 0.5-2 seconds (fast, CPU-optimized)
- **DeepSeek**: Typically 2-5 seconds (slower, more accurate)
- First DeepSeek request may be slower (model loading)

### Text Length
- **Higher isn't always better**: Depends on invoice complexity
- **DeepSeek often longer**: Better at preserving structure/formatting
- **Compare with actual invoice**: Manual verification recommended

### Success Rates
- **Target**: >90% success rate for clean invoices
- **Common failures**: Rotated images, very low quality, corrupted files
- **Service differences**: DeepSeek may handle complex layouts better

---

## Troubleshooting

### "Connection refused" Error
```bash
# Check if services are running
docker-compose ps

# Start services if needed
docker-compose up -d

# Wait for DeepSeek to be ready (2-5 minutes first time)
docker-compose logs -f deepseek
```

### "Timeout" Error
- DeepSeek can take 5-10 seconds per image
- Try increasing timeout in script (line with `timeout=60`)
- Check Docker logs: `docker-compose logs deepseek`

### "Image not found" Error
- Check path is correct (use forward slashes or double backslashes on Windows)
- Use absolute path or path relative to project root
- Example: `data/datasets/invoices_raw/batch1-0001.jpg`

---

## Results Location

All test results are saved to: `data/test_results/`

Files include:
- `dataset_validation_TIMESTAMP.json` - Dataset validation report
- `batch_test_TIMESTAMP.json` - Batch test results
- `comparison_FILENAME_TIMESTAMP.json` - Detailed comparisons

---

## Next Steps

After running these scripts:

1. **Document findings** in a simple text file or spreadsheet
2. **Identify patterns**: Which types of invoices fail? Which service is better?
3. **Select samples**: Pick 15-20 diverse invoices for deeper testing
4. **Create ground truth** (optional): Manually label key fields for accuracy testing

See `PHASE1.5_QUICKSTART.md` for more details and advanced workflows.
