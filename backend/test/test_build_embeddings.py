# test_build_embeddings.py
# Unit and integration tests for the build_embeddings module

import sys
from pathlib import Path

# Add parent directory to path to import application modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from backend.src.client_app.build_embeddings import build


class TestBuildEmbeddings:
    """Test suite for build_embeddings module"""

    @pytest.fixture
    def sample_csv_data(self):
        """Fixture providing sample CSV data for testing"""
        return pd.DataFrame({
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'text': ['Text content 1', 'Text content 2', 'Text content 3'],
            'domain': ['science', 'technology', 'health'],
            'source': ['source1.com', 'source2.com', 'source3.com']
        })

    @pytest.fixture
    def temp_dir(self):
        """Fixture providing a temporary directory for test outputs"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def temp_csv(self, sample_csv_data, temp_dir):
        """Fixture creating a temporary CSV file with sample data"""
        csv_path = Path(temp_dir) / 'test_data.csv'
        sample_csv_data.to_csv(csv_path, index=False, encoding='latin1')
        return str(csv_path)

    # CRITICAL TESTS - Core functionality

    @patch('src.build_embeddings.Embeddings')
    def test_build_creates_embeddings_successfully(self, mock_embeddings_class, temp_csv, temp_dir):
        """
        Test that build() successfully creates and saves embeddings from a valid CSV
        Priority: CRITICAL
        """
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Execute
        build(csv_path=temp_csv, out_path=out_path, encoding='latin1')
        
        # Verify embeddings were configured correctly
        call_args = mock_embeddings_class.call_args
        config = call_args[0][0] if call_args[0] else call_args[1]
        
        assert config['path'] == 'sentence-transformers/all-MiniLM-L6-v2'
        assert config['content'] is True
        assert config['scoring']['method'] == 'bm25'
        assert config['scoring']['terms'] is True
        assert config['function'] == 'sentence'
        
        # Verify index was called with records
        mock_embeddings_instance.index.assert_called_once()
        records = mock_embeddings_instance.index.call_args[0][0]
        assert len(records) == 3  # Should have 3 records from fixture
        
        # Verify save and close were called
        mock_embeddings_instance.save.assert_called_once_with(out_path)
        mock_embeddings_instance.close.assert_called_once()

    @patch('src.build_embeddings.Embeddings')
    def test_build_removes_unnamed_columns(self, mock_embeddings_class, temp_dir):
        """
        Test that build() correctly removes 'Unnamed:' columns from CSV
        Priority: CRITICAL
        """
        # Create CSV with unnamed columns
        csv_path = Path(temp_dir) / 'test_unnamed.csv'
        df = pd.DataFrame({
            'title': ['Title 1'],
            'text': ['Text 1'],
            'domain': ['science'],
            'source': ['source.com'],
            'Unnamed: 0': [0],
            'Unnamed: 1': [1]
        })
        df.to_csv(csv_path, index=False, encoding='latin1')
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Execute - should not raise error despite unnamed columns
        build(csv_path=str(csv_path), out_path=out_path)
        
        # Verify it still created the record
        mock_embeddings_instance.index.assert_called_once()

    @patch('src.build_embeddings.Embeddings')
    def test_build_drops_rows_with_missing_values(self, mock_embeddings_class, temp_dir):
        """
        Test that build() drops rows with missing required fields
        Priority: CRITICAL
        """
        # Create CSV with missing values
        csv_path = Path(temp_dir) / 'test_missing.csv'
        df = pd.DataFrame({
            'title': ['Title 1', None, 'Title 3'],
            'text': ['Text 1', 'Text 2', None],
            'domain': ['science', 'tech', 'health'],
            'source': ['source1.com', 'source2.com', 'source3.com']
        })
        df.to_csv(csv_path, index=False, encoding='latin1')
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Execute
        build(csv_path=str(csv_path), out_path=out_path)
        
        # Verify only 1 record was indexed (rows 2 and 3 should be dropped)
        records = mock_embeddings_instance.index.call_args[0][0]
        assert len(records) == 1

    @patch('src.build_embeddings.Embeddings')
    def test_build_normalizes_domain_field(self, mock_embeddings_class, temp_csv, temp_dir):
        """
        Test that build() normalizes domain field (strip + lowercase)
        Priority: IMPORTANT
        """
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Execute
        build(csv_path=temp_csv, out_path=out_path)
        
        # Verify domain normalization
        records = mock_embeddings_instance.index.call_args[0][0]
        for idx, record in records:
            assert record['domain'] == record['domain'].strip().lower()

    # IMPORTANT TESTS - Edge cases and error handling

    @patch('src.build_embeddings.Embeddings')
    def test_build_enables_graph_when_dependencies_available(
        self, mock_embeddings_class, temp_csv, temp_dir
    ):
        """
        Test that build() enables graph configuration when optional dependencies are available
        Priority: IMPORTANT
        """
        import sys
        # Mock the optional dependencies in sys.modules
        sys.modules['networkx'] = MagicMock()
        sys.modules['grandcypher'] = MagicMock()
        
        try:
            mock_embeddings_instance = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            
            out_path = str(Path(temp_dir) / 'embeddings')
            
            # Execute
            build(csv_path=temp_csv, out_path=out_path)
            
            # Verify graph config was included
            config = mock_embeddings_class.call_args[0][0]
            assert 'graph' in config
            assert config['graph'] == {'copyattributes': ['domain', 'source']}
        finally:
            # Cleanup
            if 'networkx' in sys.modules:
                del sys.modules['networkx']
            if 'grandcypher' in sys.modules:
                del sys.modules['grandcypher']

    @patch('src.build_embeddings.Embeddings')
    def test_build_works_without_graph_dependencies(self, mock_embeddings_class, temp_csv, temp_dir):
        """
        Test that build() still works when graph dependencies are not available
        Priority: IMPORTANT
        """
        import sys
        # Ensure networkx and grandcypher are NOT in sys.modules
        networkx_backup = sys.modules.pop('networkx', None)
        grandcypher_backup = sys.modules.pop('grandcypher', None)
        
        try:
            mock_embeddings_instance = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            
            out_path = str(Path(temp_dir) / 'embeddings')
            
            # Execute - should work even without graph dependencies
            build(csv_path=temp_csv, out_path=out_path)
            
            # Should still work, just without graph config
            mock_embeddings_instance.index.assert_called_once()
            mock_embeddings_instance.save.assert_called_once()
        finally:
            # Restore original state
            if networkx_backup is not None:
                sys.modules['networkx'] = networkx_backup
            if grandcypher_backup is not None:
                sys.modules['grandcypher'] = grandcypher_backup

    @patch('src.build_embeddings.Embeddings')
    def test_build_creates_correct_record_structure(self, mock_embeddings_class, temp_csv, temp_dir):
        """
        Test that build() creates records with correct structure and fields
        Priority: CRITICAL
        """
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Execute
        build(csv_path=temp_csv, out_path=out_path)
        
        # Verify record structure
        records = mock_embeddings_instance.index.call_args[0][0]
        for idx, record in records:
            assert isinstance(idx, int)
            assert isinstance(record, dict)
            assert 'text' in record
            assert 'title' in record
            assert 'domain' in record
            assert 'source' in record
            assert isinstance(record['text'], str)
            assert isinstance(record['title'], str)
            assert isinstance(record['domain'], str)
            assert isinstance(record['source'], str)

    # NICE-TO-HAVE TESTS - Additional coverage

    @patch('src.build_embeddings.Embeddings')
    def test_build_handles_different_encodings(self, mock_embeddings_class, temp_dir):
        """
        Test that build() respects the encoding parameter
        Priority: NICE-TO-HAVE
        """
        # Create CSV with special characters
        csv_path = Path(temp_dir) / 'test_encoding.csv'
        df = pd.DataFrame({
            'title': ['Título 1'],
            'text': ['Tëxt with spëcial chârs'],
            'domain': ['science'],
            'source': ['source.com']
        })
        df.to_csv(csv_path, index=False, encoding='latin1')
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Should not raise encoding errors
        build(csv_path=str(csv_path), out_path=out_path, encoding='latin1')
        
        mock_embeddings_instance.index.assert_called_once()

    @patch('src.build_embeddings.Embeddings')
    def test_build_with_empty_csv(self, mock_embeddings_class, temp_dir):
        """
        Test that build() handles empty CSV gracefully
        Priority: NICE-TO-HAVE
        """
        csv_path = Path(temp_dir) / 'empty.csv'
        df = pd.DataFrame(columns=['title', 'text', 'domain', 'source'])
        df.to_csv(csv_path, index=False, encoding='latin1')
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Execute
        build(csv_path=str(csv_path), out_path=out_path)
        
        # Should create empty index
        records = mock_embeddings_instance.index.call_args[0][0]
        assert len(records) == 0

    def test_build_missing_csv_file(self, temp_dir):
        """
        Test that build() raises appropriate error for missing CSV file
        Priority: IMPORTANT
        """
        out_path = str(Path(temp_dir) / 'embeddings')
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            build(csv_path='nonexistent.csv', out_path=out_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
