import PyPDF2

pdf = PyPDF2.PdfReader("data/acts/Motor_Vehicles_Act_1988.pdf")
print(f"Pages: {len(pdf.pages)}")
print(f"First page text: {pdf.pages[0].extract_text()[:500]}")