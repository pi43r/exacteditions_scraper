# Exacteditions Downloader

Download magazines from Exacteditions and convert them to PDF and EPUB formats.

## Installation

### Option 1: Using uv (Recommended)

```bash
uv sync
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Single Magazine Download

Interactive mode:

```bash
python main.py
```

Command line:

```bash
python main.py "Magazine Name" "https://cloudfront-url.../page-000001.webp?..."
```

### Batch Processing

Download multiple magazines from a CSV file:

```bash
python main.py --batch magazines.csv
```

#### CSV Format

Create a `magazines.csv` file with one magazine per line:

```csv
Magazine Name 1,https://cloudfront-url.../page-000001.webp?...
Magazine Name 2,https://cloudfront-url.../page-000001.webp?...
# Lines starting with # are ignored
,https://invalid-url  # Skipped (invalid)
```

## Getting the CloudFront URL

1. Open the magazine on exacteditions.com
2. Right-click on any page
3. Select **"Copy image link"**
4. Paste the URL when prompted or add to CSV file

## Output

Downloads are saved to `output/{magazine_name}/`:

- `{magazine_name}.pdf` - PDF with lossy JPEG compression (quality 90)
- `{magazine_name}.epub` - EPUB with original WebP images
- `imgs/` - Original WebP page images (deleted by default)

## Options

### Single Download

```bash
python main.py "Magazine" "URL" [OPTIONS]
```

### Batch Processing

```bash
python main.py --batch magazines.csv [OPTIONS]
```

### Available Options

- `--workers N` - Number of parallel downloads (default: 5)
- `--output {pdf|epub|both|none}` - Output format (default: both)
  - `pdf` - Generate only PDF
  - `epub` - Generate only EPUB
  - `both` - Generate both PDF and EPUB
  - `none` - Download images only, no conversion
- `--keep-images` - Keep WebP images folder after conversion (default: delete)
- `-v, --verbose` - Enable verbose logging

### Examples

```bash
# Download one magazine, keep images
python main.py "Magazine" "URL" --keep-images

# Batch download, PDF only
python main.py --batch magazines.csv --output pdf

# Batch download with 10 workers
python main.py --batch magazines.csv --workers 10

# Verbose logging
python main.py "Magazine" "URL" -v
```
