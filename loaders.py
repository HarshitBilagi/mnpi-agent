import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt

def load_document(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() for page in reader.pages)
    elif path.lower().endswith(".docx"):
        text = docx2txt.process(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def chunk_text(text, chunk_size=1500, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)
