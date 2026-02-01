"""Unit tests for DatasetLoader."""
import os
import tempfile
import shutil
import pytest
from src.dataset_loader import DatasetLoader


class TestDatasetLoader:
    """Tests for DatasetLoader"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_directory_creation(self, temp_dataset_dir):
        """Test that DatasetLoader creates directory if it doesn't exist"""
        new_dir = os.path.join(temp_dataset_dir, "new_datasets")
        assert not os.path.exists(new_dir)
        
        loader = DatasetLoader(new_dir)
        assert os.path.exists(new_dir)
    
    def test_load_txt_file(self, temp_dataset_dir):
        """Test loading .txt file"""
        txt_file = os.path.join(temp_dataset_dir, "test.txt")
        test_content = "Это тестовый текст.\nВторая строка."
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        loader = DatasetLoader(temp_dataset_dir)
        content = loader.load_txt_file(txt_file)
        
        assert content == test_content
    
    def test_empty_txt_file(self, temp_dataset_dir):
        """Test handling of empty .txt file"""
        txt_file = os.path.join(temp_dataset_dir, "empty.txt")
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        loader = DatasetLoader(temp_dataset_dir)
        content = loader.load_txt_file(txt_file)
        
        assert content == ""
    
    def test_get_supported_files(self, temp_dataset_dir):
        """Test getting list of supported files"""
        # Create various files
        with open(os.path.join(temp_dataset_dir, "file1.txt"), 'w') as f:
            f.write("test")
        with open(os.path.join(temp_dataset_dir, "file2.pdf"), 'w') as f:
            f.write("test")
        with open(os.path.join(temp_dataset_dir, "file3.docx"), 'w') as f:
            f.write("test")
        with open(os.path.join(temp_dataset_dir, "file4.xyz"), 'w') as f:
            f.write("test")
        
        loader = DatasetLoader(temp_dataset_dir)
        supported = loader.get_supported_files()
        
        assert len(supported) == 3
        assert any('file1.txt' in f for f in supported)
        assert any('file2.pdf' in f for f in supported)
        assert any('file3.docx' in f for f in supported)
        assert not any('file4.xyz' in f for f in supported)
    
    def test_load_all_datasets_txt_only(self, temp_dataset_dir):
        """Test loading all datasets with only txt files"""
        with open(os.path.join(temp_dataset_dir, "file1.txt"), 'w', encoding='utf-8') as f:
            f.write("Первый файл")
        with open(os.path.join(temp_dataset_dir, "file2.txt"), 'w', encoding='utf-8') as f:
            f.write("Второй файл")
        
        loader = DatasetLoader(temp_dataset_dir)
        combined, errors = loader.load_all_datasets()
        
        assert "Первый файл" in combined
        assert "Второй файл" in combined
        assert len(errors) == 0
    
    def test_unsupported_format_skipped(self, temp_dataset_dir):
        """Test that unsupported formats are skipped"""
        with open(os.path.join(temp_dataset_dir, "file.txt"), 'w', encoding='utf-8') as f:
            f.write("Поддерживаемый файл")
        with open(os.path.join(temp_dataset_dir, "file.xyz"), 'w', encoding='utf-8') as f:
            f.write("Неподдерживаемый файл")
        
        loader = DatasetLoader(temp_dataset_dir)
        combined, errors = loader.load_all_datasets()
        
        assert "Поддерживаемый файл" in combined
        assert "Неподдерживаемый файл" not in combined
        assert len(errors) == 0
