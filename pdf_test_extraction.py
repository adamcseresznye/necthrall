from agents.acquisition import PDFTextExtractor

if __name__ == "__main__":

    file_path = r"C:\Users\s0212777\Downloads\Allergy - 2024 - De Paepe - Integrated gut metabolome and microbiome fingerprinting reveals that dysbiosis precedes.pdf"

    with open(file_path, "rb") as file:
        pdf_bytes = file.read()

    extractor = PDFTextExtractor()

    extracted_file = extractor.extract(paper_id="test", pdf_content=pdf_bytes)
    print(extracted_file.paper_id)
    print(extracted_file.page_count)
    print(extracted_file.char_count)
    print(extracted_file.raw_text)
