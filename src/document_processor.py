import PyPDF2
import os
from typing import List
import re

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def process_documents(self, data_folder: str) -> List[str]:
        """Process all PDF documents in the data folder."""
        all_chunks = []
        
        for filename in os.listdir(data_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(data_folder, filename)
                print(f"Processing: {filename}")
                
                text = self.extract_text_from_pdf(pdf_path)
                cleaned_text = self.clean_text(text)
                chunks = self.chunk_text(cleaned_text)
                all_chunks.extend(chunks)
        
        return all_chunks
