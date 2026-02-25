"""Convert markdown to PDF using fpdf2."""
from fpdf import FPDF
from pathlib import Path
import re
import sys

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'TRANS System - First Principles', align='C')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


def convert_md_to_pdf(md_path: Path, pdf_path: Path):
    """Convert a markdown file to PDF."""
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Read markdown
    md_content = md_path.read_text(encoding='utf-8')

    # Process line by line
    lines = md_content.split('\n')
    in_code_block = False

    for line in lines:
        # Code blocks
        if line.startswith('```'):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            pdf.set_font('Courier', '', 7)
            pdf.set_fill_color(240, 240, 240)
            # Replace problematic characters and truncate
            line = line.replace('\t', '    ')
            if len(line) > 80:
                line = line[:77] + '...'
            try:
                if line.strip():
                    pdf.multi_cell(0, 4, line.encode('latin-1', 'replace').decode('latin-1'), fill=True)
                else:
                    pdf.ln(2)
            except Exception:
                pdf.ln(2)  # Skip problematic lines
            continue

        # Headers
        if line.startswith('# '):
            pdf.set_font('Helvetica', 'B', 18)
            pdf.set_text_color(30, 64, 175)
            pdf.ln(5)
            pdf.multi_cell(0, 10, line[2:])
            pdf.ln(3)
            continue

        if line.startswith('## '):
            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(37, 99, 235)
            pdf.ln(8)
            pdf.multi_cell(0, 8, line[3:])
            pdf.ln(2)
            continue

        if line.startswith('### '):
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(59, 130, 246)
            pdf.ln(5)
            pdf.multi_cell(0, 7, line[4:])
            pdf.ln(2)
            continue

        # Horizontal rule
        if line.startswith('---'):
            pdf.ln(5)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            continue

        # Blockquote
        if line.startswith('> '):
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_text_color(30, 64, 175)
            pdf.set_fill_color(239, 246, 255)
            pdf.multi_cell(0, 6, line[2:], fill=True)
            pdf.set_text_color(0, 0, 0)
            continue

        # Tables - detect by pipe character
        if '|' in line and not line.startswith('|--'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells:
                pdf.set_font('Helvetica', '', 8)
                pdf.set_text_color(0, 0, 0)
                col_width = 190 / len(cells)
                for cell in cells:
                    # Clean markdown formatting
                    cell = re.sub(r'\*\*(.*?)\*\*', r'\1', cell)
                    cell = re.sub(r'`(.*?)`', r'\1', cell)
                    # Handle encoding and truncate
                    cell = cell.encode('latin-1', 'replace').decode('latin-1')
                    display = cell[:30] + '..' if len(cell) > 30 else cell
                    try:
                        pdf.cell(col_width, 5, display, border=1)
                    except Exception:
                        pdf.cell(col_width, 5, '...', border=1)
                pdf.ln()
            continue

        # Skip table separator lines
        if line.startswith('|--'):
            continue

        # Regular text
        if line.strip():
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(51, 51, 51)
            # Clean markdown formatting
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            clean_line = re.sub(r'`(.*?)`', r'\1', clean_line)
            # Handle encoding
            clean_line = clean_line.encode('latin-1', 'replace').decode('latin-1')
            try:
                pdf.multi_cell(0, 6, clean_line)
            except Exception:
                pass  # Skip problematic lines
        else:
            pdf.ln(3)

    # Save
    pdf.output(str(pdf_path))
    print(f'PDF saved to: {pdf_path.absolute()}')


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    md_path = base_dir / 'docs' / 'SYSTEM_FIRST_PRINCIPLES.md'
    pdf_path = base_dir / 'docs' / 'SYSTEM_FIRST_PRINCIPLES.pdf'

    convert_md_to_pdf(md_path, pdf_path)
