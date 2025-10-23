#!/usr/bin/env python3

import argparse
import os
import sys
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import requests
from PIL import Image
from io import BytesIO
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from ebooklib import epub

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ExacteditionsDownloader:
    def __init__(self, magazine_name: str, cloudfront_url: str, max_workers: int = 5, output_format: str = 'both'):
        self.magazine_name = magazine_name
        self.cloudfront_url = cloudfront_url
        self.images = {}
        self.output_dir = Path(f"output/{magazine_name}")
        self.imgs_dir = self.output_dir / "imgs"
        self.max_workers = max_workers
        self.output_format = output_format
        self.images_lock = Lock()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.imgs_dir.mkdir(parents=True, exist_ok=True)

    def parse_url(self) -> dict:
        parsed = urlparse(self.cloudfront_url)
        
        path_match = re.search(r'page-(\d+)\.webp', parsed.path)
        if not path_match:
            raise ValueError("Could not extract page number from URL")
        
        base_path = parsed.path.rsplit('/', 1)[0]
        
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        query_params_flat = {k: v[0] if v else '' for k, v in query_params.items()}
        
        return {
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'base_path': base_path,
            'query_params': query_params_flat,
        }

    def build_page_url(self, page_num: int, url_info: dict) -> str:
        page_str = f"{page_num:06d}"
        path = f"{url_info['base_path']}/page-{page_str}.webp"
        
        query_string = urlencode(url_info['query_params'])
        
        url = urlunparse((
            url_info['scheme'],
            url_info['netloc'],
            path,
            '',
            query_string,
            ''
        ))
        return url

    def image_exists(self, page_num: int) -> bool:
        filename = self.imgs_dir / f"page-{page_num:06d}.webp"
        return filename.exists()

    def fetch_page(self, page_num: int, url_info: dict) -> tuple:
        if self.image_exists(page_num):
            logger.info(f"Page {page_num}: Already downloaded, skipping")
            with open(self.imgs_dir / f"page-{page_num:06d}.webp", 'rb') as f:
                return (page_num, f.read(), True)
        
        url = self.build_page_url(page_num, url_info)
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 404:
                logger.info(f"Page {page_num}: 404 Not Found")
                return (page_num, None, False)
            
            if response.status_code != 200:
                logger.warning(f"Page {page_num}: HTTP {response.status_code}")
                return (page_num, None, False)
            
            logger.info(f"Page {page_num}: Downloaded ({len(response.content)} bytes)")
            return (page_num, response.content, True)
            
        except requests.RequestException as e:
            logger.warning(f"Page {page_num}: Request failed - {e}")
            return (page_num, None, False)

    def scrape(self) -> None:
        logger.info(f"Parsing URL: {self.cloudfront_url}")
        url_info = self.parse_url()
        logger.info(f"Starting from page 1 with {self.max_workers} parallel workers")
        
        page_num = 1
        consecutive_failures = 0
        max_failures = 3
        last_success = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            while consecutive_failures < max_failures:
                while len(futures) < self.max_workers and consecutive_failures < max_failures:
                    future = executor.submit(self.fetch_page, page_num, url_info)
                    futures[future] = page_num
                    page_num += 1
                
                for future in as_completed(futures):
                    page_num_result, content, success = future.result()
                    del futures[future]
                    
                    if success:
                        with self.images_lock:
                            self.images[page_num_result] = content
                        last_success = page_num_result
                        consecutive_failures = 0
                    else:
                        if page_num_result > last_success:
                            consecutive_failures += 1
                    
                    if consecutive_failures >= max_failures:
                        break
        
        if not self.images:
            raise RuntimeError("No images were downloaded")
        
        logger.info(f"Downloaded {len(self.images)} pages")

    def save_webp_images(self) -> None:
        logger.info(f"Saving WebP images to {self.imgs_dir}")
        for page_num in sorted(self.images.keys()):
            filename = self.imgs_dir / f"page-{page_num:06d}.webp"
            if not filename.exists():
                with open(filename, 'wb') as f:
                    f.write(self.images[page_num])

    def convert_webp_to_pil(self, webp_data: bytes) -> Image.Image:
        return Image.open(BytesIO(webp_data)).convert('RGB')

    def generate_pdf(self) -> bytes:
        logger.info("Generating PDF...")
        
        sorted_pages = sorted(self.images.keys())
        pil_images = [self.convert_webp_to_pil(self.images[page_num]) for page_num in sorted_pages]
        
        pdf_buffer = BytesIO()
        
        first_image = pil_images[0]
        remaining_images = pil_images[1:]
        
        first_image.save(
            pdf_buffer,
            format='PDF',
            save_all=True,
            append_images=remaining_images,
            quality=90
        )
        
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    def generate_epub(self) -> bytes:
        logger.info("Generating EPUB...")
        
        book = epub.EpubBook()
        book.set_identifier(f'exacteditions-{self.magazine_name}')
        book.set_title(self.magazine_name)
        book.set_language('en')
        
        sorted_pages = sorted(self.images.keys())
        
        if not sorted_pages:
            raise RuntimeError("No pages to generate EPUB")
        
        cover_page_num = sorted_pages[0]
        cover_webp = self.images[cover_page_num]
        book.set_cover('cover.webp', cover_webp, create_page=True)
        
        chapters = []
        
        for idx, page_num in enumerate(sorted_pages[1:], 1):
            webp_data = self.images[page_num]
            
            img_filename = f'page-{idx:06d}.webp'
            img_item = epub.EpubImage(
                uid=f'img_{idx:06d}',
                file_name=img_filename,
                media_type='image/webp',
                content=webp_data
            )
            book.add_item(img_item)
            
            chapter = epub.EpubHtml(
                title='',
                file_name=f'page-{idx:06d}.xhtml',
                uid=f'page_{idx:06d}'
            )
            chapter.content = f'<html><body style="margin:0;padding:0"><img src="{img_filename}" style="width:100%;height:auto" /></body></html>'
            book.add_item(chapter)
            chapters.append(chapter)
        
        # book.toc = ()
        book.spine = ['nav'] + chapters
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        epub_buffer = BytesIO()
        epub.write_epub(epub_buffer, book, {})
        
        epub_buffer.seek(0)
        return epub_buffer.getvalue()

    def save_outputs(self) -> None:
        if self.output_format in ('pdf', 'both'):
            pdf_filename = self.output_dir / f"{self.magazine_name}.pdf"
            logger.info(f"Saving PDF to {pdf_filename}")
            with open(pdf_filename, 'wb') as f:
                f.write(self.generate_pdf())
        
        if self.output_format in ('epub', 'both'):
            epub_filename = self.output_dir / f"{self.magazine_name}.epub"
            logger.info(f"Saving EPUB to {epub_filename}")
            with open(epub_filename, 'wb') as f:
                f.write(self.generate_epub())
        
        logger.info("Done!")

    def run(self) -> None:
        self.scrape()
        self.save_webp_images()
        if self.output_format != 'none':
            self.save_outputs()
        else:
            logger.info("Output format is 'none', skipping PDF/EPUB generation")


def interactive_mode():
    print("\n=== Exacteditions Downloader ===\n")
    
    magazine_name = input("Enter magazine name: ").strip()
    if not magazine_name:
        logger.error("Magazine name cannot be empty")
        sys.exit(1)
    
    cloudfront_url = input("Enter CloudFront image URL: ").strip()
    if not cloudfront_url:
        logger.error("URL cannot be empty")
        sys.exit(1)
    
    return magazine_name, cloudfront_url


def main():
    parser = argparse.ArgumentParser(
        description='Download magazines from Exacteditions and convert to PDF/EPUB'
    )
    parser.add_argument(
        'magazine_name',
        nargs='?',
        help='Magazine name'
    )
    parser.add_argument(
        'url',
        nargs='?',
        help='CloudFront image URL from exacteditions.com'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel download workers (default: 5)'
    )
    parser.add_argument(
        '--output',
        choices=['pdf', 'epub', 'both', 'none'],
        default='both',
        help='Output format: pdf, epub, both, or none (default: both)'
    )
    
    args = parser.parse_args()
    
    if args.magazine_name and args.url:
        magazine_name = args.magazine_name
        cloudfront_url = args.url
    else:
        magazine_name, cloudfront_url = interactive_mode()
    
    try:
        downloader = ExacteditionsDownloader(magazine_name, cloudfront_url, max_workers=args.workers, output_format=args.output)
        downloader.run()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
