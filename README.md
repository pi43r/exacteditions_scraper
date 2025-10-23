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

```bash
python main.py "Magazine Name" "https://cloudfront-url.../page-000001.webp?..."
```

Or run interactively:

```bash
python main.py
```

## Getting the CloudFront URL

1. Open the magazine on exacteditions.com
2. Right-click on any page
3. Select **"Copy image link"** or **"Copy link"**
4. Paste the URL when prompted

## Output

Downloads are saved to `output/{magazine_name}/`:

- `{magazine_name}.pdf` - PDF with lossy JPEG compression (quality 90)
- `{magazine_name}.epub` - EPUB with individual pages for device reading
- `imgs/` - Original WebP page images

## Options

```bash
python main.py "Magazine Name" "URL" --workers 5 --output both
```

- `--workers` - Number of parallel downloads (default: 5)
- `--output` - Output format:
  - `pdf` - Generate only PDF
  - `epub` - Generate only EPUB
  - `both` - Generate both PDF and EPUB (default)
  - `none` - Download images only, no PDF/EPUB generation
