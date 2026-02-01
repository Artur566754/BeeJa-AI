"""Property-based tests for DatasetLoader."""
import os
import tempfile
import shutil
from hypothesis import given, strategies as st, settings
from src.dataset_loader import DatasetLoader


class TestDatasetLoaderProperties:
    """Property-based tests for DatasetLoader
    
    Feature: custom-ai-model
    """
    
    @given(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    @settings(max_examples=100)
    def test_txt_file_extraction(self, text: str):
        """
        Property 9: Multi-Format File Support
        For any supported file format (.txt), the loader should successfully extract text
        
        Validates: Requirements 3.1, 3.2, 3.3
        """
        temp_dir = tempfile.mkdtemp()
        
        try:
            txt_file = os.path.join(temp_dir, "test.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            loader = DatasetLoader(temp_dir)
            extracted = loader.load_txt_file(txt_file)
            
            # Normalize line endings for comparison (Windows converts \r to \n)
            normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
            assert extracted == normalized_text, "Extracted text should match original (with normalized line endings)"
        finally:
            shutil.rmtree(temp_dir)
    
    @given(st.lists(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()), min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_complete_file_processing(self, texts: list):
        """
        Property 11: Complete File Processing
        For any set of supported files, all files should be processed
        
        Validates: Requirements 3.5
        """
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create multiple txt files
            for i, text in enumerate(texts):
                txt_file = os.path.join(temp_dir, f"file{i}.txt")
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            loader = DatasetLoader(temp_dir)
            combined, errors = loader.load_all_datasets()
            
            # All texts should be in combined output (with normalized line endings)
            for text in texts:
                normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
                assert normalized_text in combined, f"All files should be processed"
            
            assert len(errors) == 0, "No errors should occur with valid files"
        finally:
            shutil.rmtree(temp_dir)
    
    @given(
        st.lists(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()), min_size=1, max_size=3),
        st.integers(min_value=0, max_value=2)
    )
    @settings(max_examples=100)
    def test_error_resilience(self, texts: list, corrupt_index: int):
        """
        Property 13: Error Resilience
        For any directory with readable and corrupted files, all readable files should be processed
        
        Validates: Requirements 3.7
        """
        if not texts:
            return
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create valid txt files
            for i, text in enumerate(texts):
                txt_file = os.path.join(temp_dir, f"file{i}.txt")
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            # Create a "corrupted" file (unsupported format)
            corrupt_file = os.path.join(temp_dir, "corrupted.xyz")
            with open(corrupt_file, 'w', encoding='utf-8') as f:
                f.write("This should be skipped")
            
            loader = DatasetLoader(temp_dir)
            combined, errors = loader.load_all_datasets()
            
            # All valid texts should still be processed (with normalized line endings)
            for text in texts:
                normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
                assert normalized_text in combined, "Readable files should be processed despite corrupted files"
            
            # Corrupted file should not be in output
            assert "This should be skipped" not in combined
        finally:
            shutil.rmtree(temp_dir)
