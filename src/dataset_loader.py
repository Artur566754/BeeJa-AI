"""Dataset loader for various file formats."""
import os
import logging
from typing import List, Tuple

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyPDF2 not installed. PDF support disabled.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("python-docx not installed. DOCX support disabled.")

try:
    import textract
    TEXTRACT_AVAILABLE = True
except ImportError:
    TEXTRACT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("textract not installed. DOC support disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Загрузчик датасетов из различных форматов файлов"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx'}
    
    def __init__(self, dataset_dir: str):
        """
        Args:
            dataset_dir: Путь к директории с датасетами
        """
        self.dataset_dir = dataset_dir
        
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            logger.info(f"Created dataset directory: {dataset_dir}")
    
    def get_supported_files(self) -> List[str]:
        """
        Получение списка всех поддерживаемых файлов в директории
        
        Returns:
            Список путей к файлам
        """
        supported_files = []
        
        if not os.path.exists(self.dataset_dir):
            return supported_files
        
        for filename in os.listdir(self.dataset_dir):
            filepath = os.path.join(self.dataset_dir, filename)
            
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filename)
                if ext.lower() in self.SUPPORTED_EXTENSIONS:
                    supported_files.append(filepath)
        
        return supported_files
    
    def load_txt_file(self, filepath: str) -> str:
        """
        Загрузка .txt файла
        
        Args:
            filepath: Путь к .txt файлу
            
        Returns:
            Текст из файла или пустая строка если файл пустой
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Handle empty files gracefully
            if not content.strip():
                logger.warning(f"Empty file: {filepath}")
                return ""
            
            return content
        except Exception as e:
            logger.error(f"Error loading txt file {filepath}: {e}")
            raise
    
    def load_pdf_file(self, filepath: str) -> str:
        """
        Загрузка .pdf файла с использованием PyPDF2
        
        Args:
            filepath: Путь к .pdf файлу
            
        Returns:
            Извлеченный текст из PDF
        """
        if not PDF_AVAILABLE:
            logger.error(f"Cannot load PDF {filepath}: PyPDF2 not installed")
            raise ImportError("PyPDF2 is required for PDF support")
        
        try:
            reader = PdfReader(filepath)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            combined_text = '\n'.join(text_parts)
            
            if not combined_text.strip():
                logger.warning(f"No text extracted from PDF: {filepath}")
                return ""
            
            return combined_text
        except Exception as e:
            logger.error(f"Error loading PDF file {filepath}: {e}")
            raise
    
    def load_docx_file(self, filepath: str) -> str:
        """
        Загрузка .docx файла с использованием python-docx
        
        Args:
            filepath: Путь к .docx файлу
            
        Returns:
            Извлеченный текст из Word документа
        """
        if not DOCX_AVAILABLE:
            logger.error(f"Cannot load DOCX {filepath}: python-docx not installed")
            raise ImportError("python-docx is required for DOCX support")
        
        try:
            doc = Document(filepath)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            combined_text = '\n'.join(paragraphs)
            
            if not combined_text.strip():
                logger.warning(f"No text extracted from DOCX: {filepath}")
                return ""
            
            return combined_text
        except Exception as e:
            logger.error(f"Error loading DOCX file {filepath}: {e}")
            raise
    
    def load_doc_file(self, filepath: str) -> str:
        """
        Загрузка .doc файла (legacy format)
        
        Args:
            filepath: Путь к .doc файлу
            
        Returns:
            Извлеченный текст из старого формата Word
        """
        if not TEXTRACT_AVAILABLE:
            logger.error(f"Cannot load DOC {filepath}: textract not installed")
            raise ImportError("textract is required for DOC support")
        
        try:
            text = textract.process(filepath).decode('utf-8')
            
            if not text.strip():
                logger.warning(f"No text extracted from DOC: {filepath}")
                return ""
            
            return text
        except Exception as e:
            logger.error(f"Error loading DOC file {filepath}: {e}")
            raise
    
    def load_all_datasets(self) -> Tuple[str, List[str]]:
        """
        Загрузка всех поддерживаемых файлов из директории
        
        Returns:
            Tuple of (combined_text, error_list)
        """
        all_text = []
        errors = []
        
        supported_files = self.get_supported_files()
        
        if not supported_files:
            logger.warning(f"No supported files found in {self.dataset_dir}")
            return "", []
        
        for filepath in supported_files:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            try:
                if ext == '.txt':
                    text = self.load_txt_file(filepath)
                elif ext == '.pdf':
                    text = self.load_pdf_file(filepath)
                elif ext == '.docx':
                    text = self.load_docx_file(filepath)
                elif ext == '.doc':
                    text = self.load_doc_file(filepath)
                else:
                    logger.info(f"Skipping unsupported format: {filepath}")
                    continue
                
                if text.strip():
                    all_text.append(text)
                    logger.info(f"Successfully loaded: {filepath}")
            except Exception as e:
                error_msg = f"Failed to load {filepath}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                # Continue processing other files
        
        combined_text = '\n\n'.join(all_text)
        return combined_text, errors
