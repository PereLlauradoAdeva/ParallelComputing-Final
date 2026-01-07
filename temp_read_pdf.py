
import sys
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)

try:
    from pypdf import PdfReader
    
    reader = PdfReader("elaborati_finali_Parallel_Programming.pdf")
    
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- PAGE {i+1} ---\n")
            f.write(text)
            f.write("\n------------------\n")
            
    print("Done writing to pdf_content.txt")

except Exception as e:
    print(f"ERROR: {e}")
