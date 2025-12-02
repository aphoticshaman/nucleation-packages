#!/usr/bin/env python3
"""
PDF/DOC Chunker - Extracts text from documents into manageable chunks.
Handles PDFs, DOCs, and PPTs for LLM ingestion.
"""

import os
import sys
from pathlib import Path

def extract_pdf_text(pdf_path: str, pages_per_chunk: int = 5) -> list[dict]:
    """Extract text from PDF, chunked by pages."""
    import fitz  # PyMuPDF

    chunks = []
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)
        chunk_text = []

        for page_num in range(start_page, end_page):
            page = doc[page_num]
            text = page.get_text()
            chunk_text.append(f"--- PAGE {page_num + 1} ---\n{text}")

        chunks.append({
            "source": os.path.basename(pdf_path),
            "pages": f"{start_page + 1}-{end_page}",
            "total_pages": total_pages,
            "text": "\n\n".join(chunk_text)
        })

    doc.close()
    return chunks


def extract_doc_text(doc_path: str) -> list[dict]:
    """Extract text from DOC/DOCX files."""
    # Try using antiword or catdoc for .doc files
    import subprocess

    ext = Path(doc_path).suffix.lower()

    try:
        if ext == '.doc':
            # Try antiword first, fall back to catdoc
            try:
                result = subprocess.run(['antiword', doc_path], capture_output=True, text=True)
                if result.returncode == 0:
                    text = result.stdout
                else:
                    result = subprocess.run(['catdoc', doc_path], capture_output=True, text=True)
                    text = result.stdout
            except FileNotFoundError:
                result = subprocess.run(['catdoc', doc_path], capture_output=True, text=True)
                text = result.stdout
        else:
            # For docx, we'd need python-docx
            text = f"[DOCX format - needs python-docx library]"
    except Exception as e:
        text = f"[Error extracting: {e}]"

    return [{
        "source": os.path.basename(doc_path),
        "pages": "all",
        "total_pages": 1,
        "text": text
    }]


def extract_ppt_text(ppt_path: str) -> list[dict]:
    """Extract text from PPT files using basic methods."""
    import subprocess
    import zipfile
    import re

    ext = Path(ppt_path).suffix.lower()

    # For .pptx, it's a zip of XML files
    if ext == '.pptx':
        try:
            text_parts = []
            with zipfile.ZipFile(ppt_path, 'r') as z:
                for name in z.namelist():
                    if name.startswith('ppt/slides/slide') and name.endswith('.xml'):
                        content = z.read(name).decode('utf-8', errors='ignore')
                        # Extract text between <a:t> tags
                        texts = re.findall(r'<a:t>([^<]+)</a:t>', content)
                        if texts:
                            text_parts.append(f"--- SLIDE ---\n" + "\n".join(texts))
            text = "\n\n".join(text_parts)
        except Exception as e:
            text = f"[Error extracting PPTX: {e}]"
    else:
        # .ppt (binary) - try catppt or strings
        try:
            result = subprocess.run(['catppt', ppt_path], capture_output=True, text=True)
            if result.returncode == 0:
                text = result.stdout
            else:
                # Fall back to strings command
                result = subprocess.run(['strings', ppt_path], capture_output=True, text=True)
                text = result.stdout
        except Exception as e:
            text = f"[Error extracting PPT: {e}]"

    # Chunk if text is very long
    chunks = []
    lines = text.split('\n')
    chunk_size = 500  # lines per chunk

    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunks.append({
            "source": os.path.basename(ppt_path),
            "pages": f"lines {i+1}-{i+len(chunk_lines)}",
            "total_pages": len(lines),
            "text": "\n".join(chunk_lines)
        })

    return chunks if chunks else [{"source": os.path.basename(ppt_path), "pages": "all", "total_pages": 1, "text": text}]


def process_directory(input_dir: str, output_dir: str, pages_per_chunk: int = 5):
    """Process all documents in directory, output chunks as text files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for file in input_path.iterdir():
        if file.is_file():
            ext = file.suffix.lower()
            print(f"Processing: {file.name} ({file.stat().st_size / 1024:.1f} KB)")

            try:
                if ext == '.pdf':
                    chunks = extract_pdf_text(str(file), pages_per_chunk)
                elif ext in ['.doc', '.docx']:
                    chunks = extract_doc_text(str(file))
                elif ext in ['.ppt', '.pptx']:
                    chunks = extract_ppt_text(str(file))
                else:
                    print(f"  Skipping unsupported format: {ext}")
                    continue

                # Write chunks to output files
                base_name = file.stem
                for i, chunk in enumerate(chunks):
                    chunk_file = output_path / f"{base_name}_chunk_{i+1:03d}.txt"
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        f.write(f"SOURCE: {chunk['source']}\n")
                        f.write(f"PAGES: {chunk['pages']} of {chunk['total_pages']}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(chunk['text'])

                print(f"  Created {len(chunks)} chunk(s)")
                results.append({"file": file.name, "chunks": len(chunks)})

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"file": file.name, "error": str(e)})

    return results


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./ref_docs_READ_ANALYZE_THEN_DELETE"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./ref_docs_chunks"
    pages_per_chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # Smaller chunks for safety

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Pages per chunk: {pages_per_chunk}")
    print("-" * 60)

    results = process_directory(input_dir, output_dir, pages_per_chunk)

    print("-" * 60)
    print("SUMMARY:")
    for r in results:
        if "error" in r:
            print(f"  {r['file']}: ERROR - {r['error']}")
        else:
            print(f"  {r['file']}: {r['chunks']} chunks")
