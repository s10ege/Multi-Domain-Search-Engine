# test_search_engine.py
# Unit and integration tests for the search_engine module

import sys
from pathlib import Path

# Add parent directory to path to import application modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import tempfile
import shutil

from src.search_engine import (
    SearchResources,
    load_resources,
    close_resources,
    _domain_clause,
    _search_candidates,
    search,
    rag_answer,
    _normalize_rag_answer,
    summarize_results,
    more_like_this,
)


class TestSearchResources:
    """Test suite for SearchResources dataclass"""

    def test_search_resources_is_frozen(self):
        """
        Test that SearchResources is immutable (frozen dataclass)
        Priority: IMPORTANT
        """
        mock_embeddings = Mock()
        mock_rag = Mock()
        resources = SearchResources(
            titles=['Title 1'],
            texts=['Text 1'],
            domains=['science'],
            sources=['source.com'],
            embeddings=mock_embeddings,
            rag=mock_rag
        )
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            resources.titles = ['New Title']


class TestLoadResources:
    """Test suite for load_resources function"""

    @pytest.fixture
    def temp_dir(self):
        """Fixture providing a temporary directory"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def sample_csv(self, temp_dir):
        """Fixture creating a sample CSV file"""
        csv_path = Path(temp_dir) / 'test_data.csv'
        df = pd.DataFrame({
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'text': ['Text content 1', 'Text content 2', 'Text content 3'],
            'domain': ['Science', ' TECHNOLOGY ', 'health'],
            'source': ['source1.com', 'source2.com', 'source3.com'],
            'Unnamed: 0': [0, 1, 2]  # Should be dropped
        })
        df.to_csv(csv_path, index=False, encoding='latin1')
        return str(csv_path)

    @patch('src.search_engine.Embeddings')
    @patch('src.search_engine.RAG')
    @patch('src.search_engine.Path')
    def test_load_resources_success(self, mock_path_class, mock_rag_class, mock_embeddings_class, sample_csv):
        """
        Test successful loading of resources with valid CSV and embeddings
        Priority: CRITICAL
        """
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        mock_rag_instance = MagicMock()
        mock_rag_class.return_value = mock_rag_instance
        
        # Mock path existence check
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance
        
        # Execute
        resources = load_resources(csv_path=sample_csv, embeddings_path='embeddings')
        
        # Verify structure
        assert isinstance(resources, SearchResources)
        assert len(resources.titles) == 3
        assert len(resources.texts) == 3
        assert len(resources.domains) == 3
        assert len(resources.sources) == 3
        
        # Verify domain normalization (lowercase + strip)
        assert 'science' in resources.domains
        assert 'technology' in resources.domains
        assert 'health' in resources.domains
        
        # Verify embeddings configuration
        embeddings_config = mock_embeddings_class.call_args[0][0]
        assert embeddings_config['path'] == 'BAAI/bge-small-en-v1.5'
        assert embeddings_config['content'] is True
        assert embeddings_config['scoring']['method'] == 'bm25'
        assert embeddings_config['instructions']['query'].startswith('Represent this sentence')
        
        # Verify RAG initialization
        mock_rag_class.assert_called_once()
        rag_args = mock_rag_class.call_args
        assert rag_args[0][1] == 'Qwen/Qwen3-1.7B'  # Model path

    @patch('src.search_engine.Embeddings')
    @patch('src.search_engine.RAG')
    @patch('src.search_engine.Path')
    def test_load_resources_removes_unnamed_columns(self, mock_path_class, mock_rag_class, mock_embeddings_class, sample_csv):
        """
        Test that load_resources correctly removes 'Unnamed:' columns
        Priority: CRITICAL
        """
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        mock_rag_instance = MagicMock()
        mock_rag_class.return_value = mock_rag_instance
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance
        
        # Should not raise error despite Unnamed columns in CSV
        resources = load_resources(csv_path=sample_csv, embeddings_path='embeddings')
        assert len(resources.titles) == 3

    @patch('src.search_engine.Embeddings')
    @patch('src.search_engine.RAG')
    @patch('src.search_engine.Path')
    def test_load_resources_drops_missing_values(self, mock_path_class, mock_rag_class, mock_embeddings_class, temp_dir):
        """
        Test that load_resources drops rows with missing required fields
        Priority: CRITICAL
        """
        # Create CSV with missing values
        csv_path = Path(temp_dir) / 'missing.csv'
        df = pd.DataFrame({
            'title': ['Title 1', None, 'Title 3'],
            'text': ['Text 1', 'Text 2', None],
            'domain': ['science', 'tech', 'health'],
            'source': ['source1.com', None, 'source3.com']
        })
        df.to_csv(csv_path, index=False, encoding='latin1')
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        mock_rag_instance = MagicMock()
        mock_rag_class.return_value = mock_rag_instance
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance
        
        # Execute
        resources = load_resources(csv_path=str(csv_path), embeddings_path='embeddings')
        
        # Only first row should remain (others have missing values)
        assert len(resources.titles) == 1
        assert resources.titles[0] == 'Title 1'


class TestCloseResources:
    """Test suite for close_resources function"""

    def test_close_resources_calls_embeddings_close(self):
        """
        Test that close_resources properly closes embeddings
        Priority: CRITICAL
        """
        mock_embeddings = MagicMock()
        mock_rag = Mock()
        
        resources = SearchResources(
            titles=[],
            texts=[],
            domains=[],
            sources=[],
            embeddings=mock_embeddings,
            rag=mock_rag
        )
        
        close_resources(resources)
        
        mock_embeddings.close.assert_called_once()


class TestDomainClause:
    """Test suite for _domain_clause helper function"""

    def test_domain_clause_with_none(self):
        """
        Test _domain_clause returns None when domain is None
        Priority: CRITICAL
        """
        result = _domain_clause(None)
        assert result is None

    def test_domain_clause_with_string(self):
        """
        Test _domain_clause with single domain string
        Priority: CRITICAL
        """
        result = _domain_clause('science')
        assert result == "domain = 'science'"

    def test_domain_clause_with_list_single(self):
        """
        Test _domain_clause with list containing one domain
        Priority: CRITICAL
        """
        result = _domain_clause(['technology'])
        assert result == "domain = 'technology'"

    def test_domain_clause_with_list_multiple_uses_first_only(self):
        """
        Test _domain_clause with multiple domains - should only use first
        Priority: CRITICAL
        """
        result = _domain_clause(['science', 'technology', 'health'])
        assert result == "domain = 'science'"
        assert 'technology' not in result
        assert 'health' not in result

    def test_domain_clause_normalizes_domain(self):
        """
        Test _domain_clause normalizes domain (lowercase + strip)
        Priority: IMPORTANT
        """
        result = _domain_clause('  SCIENCE  ')
        assert result == "domain = 'science'"

    def test_domain_clause_escapes_quotes(self):
        """
        Test _domain_clause properly escapes single quotes for SQL safety
        Priority: IMPORTANT
        """
        result = _domain_clause("science's")
        assert result == "domain = 'science''s'"

    def test_domain_clause_with_empty_string(self):
        """
        Test _domain_clause returns None for empty string
        Priority: IMPORTANT
        """
        result = _domain_clause('')
        assert result is None

    def test_domain_clause_with_empty_list(self):
        """
        Test _domain_clause returns None for empty list
        Priority: IMPORTANT
        """
        result = _domain_clause([])
        assert result is None

    def test_domain_clause_with_whitespace_only(self):
        """
        Test _domain_clause returns None for whitespace-only string
        Priority: NICE-TO-HAVE
        """
        result = _domain_clause('   ')
        assert result is None


class TestSearchCandidates:
    """Test suite for _search_candidates helper function"""

    def test_search_candidates_basic_query(self):
        """
        Test _search_candidates executes basic search query
        Priority: CRITICAL
        """
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = [
            {'id': 0, 'score': 0.95},
            {'id': 1, 'score': 0.85}
        ]
        
        resources = SearchResources(
            titles=['Title 1', 'Title 2'],
            texts=['Text 1', 'Text 2'],
            domains=['science', 'tech'],
            sources=['source1.com', 'source2.com'],
            embeddings=mock_embeddings,
            rag=Mock()
        )
        
        results = _search_candidates(resources, 'test query', candidate_k=10)
        
        # Verify search was called
        mock_embeddings.search.assert_called_once()
        
        # Verify results structure
        assert len(results) == 2
        assert results[0]['title'] == 'Title 1'
        assert results[0]['score'] == 0.95
        assert results[0]['index'] == 0

    def test_search_candidates_with_domain_filter(self):
        """
        Test _search_candidates includes domain filter in SQL query
        Priority: CRITICAL
        """
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = []
        
        resources = SearchResources(
            titles=[],
            texts=[],
            domains=[],
            sources=[],
            embeddings=mock_embeddings,
            rag=Mock()
        )
        
        _search_candidates(resources, 'test query', candidate_k=10, domain='science')
        
        # Verify SQL query includes domain filter
        call_args = mock_embeddings.search.call_args
        sql = call_args[0][0]
        assert 'domain' in sql.lower()
        assert "domain = 'science'" in sql

    def test_search_candidates_handles_tuple_results(self):
        """
        Test _search_candidates handles tuple format results from txtai
        Priority: IMPORTANT
        """
        mock_embeddings = MagicMock()
        # Some txtai versions return tuples instead of dicts
        mock_embeddings.search.return_value = [
            (0, 0.95),
            (1, 0.85)
        ]
        
        resources = SearchResources(
            titles=['Title 1', 'Title 2'],
            texts=['Text 1', 'Text 2'],
            domains=['science', 'tech'],
            sources=['source1.com', 'source2.com'],
            embeddings=mock_embeddings,
            rag=Mock()
        )
        
        results = _search_candidates(resources, 'test query', candidate_k=10)
        
        assert len(results) == 2
        assert results[0]['score'] == 0.95


class TestSearch:
    """Test suite for search function"""

    @patch('src.search_engine._get_similarity_pipeline')
    @patch('src.search_engine._search_candidates')
    def test_search_returns_top_k_results(self, mock_search_candidates, mock_get_similarity):
        """
        Test search returns exactly top_k results
        Priority: CRITICAL
        """
        # Mock returns 10 results
        mock_results = [
            {'title': f'Title {i}', 'text': f'Text {i}', 'score': 1.0 - i*0.1}
            for i in range(10)
        ]
        mock_search_candidates.return_value = mock_results
        mock_similarity = Mock(return_value=[(i, 1.0 - i * 0.1) for i in range(10)])
        mock_get_similarity.return_value = mock_similarity
        
        resources = Mock()
        
        results = search(resources, 'test query', top_k=5)
        
        # Should return only top 5
        assert len(results) == 5

    @patch('src.search_engine._get_similarity_pipeline')
    @patch('src.search_engine._search_candidates')
    def test_search_requests_more_candidates(self, mock_search_candidates, mock_get_similarity):
        """
        Test search requests more candidates than top_k for better results
        Priority: IMPORTANT
        """
        mock_search_candidates.return_value = []
        resources = Mock()
        
        search(resources, 'test query', top_k=5)
        
        # Should request top_k * 4 candidates (or min 20)
        call_args = mock_search_candidates.call_args
        candidate_k = call_args[0][2]
        assert candidate_k >= 20

    @patch('src.search_engine._search_candidates')
    def test_search_passes_domain_filter(self, mock_search_candidates):
        """
        Test search passes domain filter to _search_candidates
        Priority: CRITICAL
        """
        mock_search_candidates.return_value = []
        resources = Mock()
        
        search(resources, 'test query', top_k=5, domain='science')
        
        call_args = mock_search_candidates.call_args
        assert call_args[1]['domain'] == 'science'


class TestNormalizeRagAnswer:
    """Test suite for _normalize_rag_answer helper function"""

    def test_normalize_empty_list(self):
        """
        Test _normalize_rag_answer handles empty list
        Priority: IMPORTANT
        """
        result = _normalize_rag_answer([])
        assert result == ""

    def test_normalize_list_with_elements(self):
        """
        Test _normalize_rag_answer extracts first element from list
        Priority: CRITICAL
        """
        result = _normalize_rag_answer(['answer text', 'other'])
        assert result == "answer text"

    def test_normalize_tuple_with_answer(self):
        """
        Test _normalize_rag_answer extracts answer from tuple
        Priority: CRITICAL
        """
        result = _normalize_rag_answer(('question', 'answer text'))
        assert result == "answer text"

    def test_normalize_single_element_tuple(self):
        """
        Test _normalize_rag_answer handles single-element tuple
        Priority: IMPORTANT
        """
        result = _normalize_rag_answer(('answer',))
        assert result == "answer"

    def test_normalize_dict_with_answer_key(self):
        """
        Test _normalize_rag_answer extracts 'answer' key from dict
        Priority: CRITICAL
        """
        result = _normalize_rag_answer({'answer': 'answer text'})
        assert result == "answer text"

    def test_normalize_dict_with_text_key(self):
        """
        Test _normalize_rag_answer fallback to 'text' key from dict
        Priority: IMPORTANT
        """
        result = _normalize_rag_answer({'text': 'answer text'})
        assert result == "answer text"

    def test_normalize_string_direct(self):
        """
        Test _normalize_rag_answer handles plain string
        Priority: CRITICAL
        """
        result = _normalize_rag_answer('answer text')
        assert result == "answer text"


class TestRagAnswer:
    """Test suite for rag_answer function"""

    @patch('src.search_engine.search')
    def test_rag_answer_returns_empty_when_no_results(self, mock_search):
        """
        Test rag_answer returns empty answer when search finds nothing
        Priority: CRITICAL
        """
        mock_search.return_value = []
        
        mock_rag = Mock()
        resources = SearchResources(
            titles=[],
            texts=[],
            domains=[],
            sources=[],
            embeddings=Mock(),
            rag=mock_rag
        )
        
        result = rag_answer(resources, 'test question')
        
        assert result['answer'] == ''
        assert result['results'] == []
        mock_rag.assert_not_called()

    @patch('src.search_engine.search')
    def test_rag_answer_calls_rag_with_context(self, mock_search):
        """
        Test rag_answer passes search results as context to RAG
        Priority: CRITICAL
        """
        mock_search.return_value = [
            {'text': 'Result text 1', 'title': 'Title 1'},
            {'text': 'Result text 2', 'title': 'Title 2'}
        ]
        
        mock_rag = MagicMock()
        mock_rag.return_value = 'Generated answer'
        
        resources = SearchResources(
            titles=[],
            texts=[],
            domains=[],
            sources=[],
            embeddings=Mock(),
            rag=mock_rag
        )
        
        result = rag_answer(resources, 'test question', top_k=2)
        
        # Verify RAG was called with question and context
        mock_rag.assert_called_once()
        call_args = mock_rag.call_args
        assert call_args[0][0] == 'test question'
        assert 'texts' in call_args[1]
        assert len(call_args[1]['texts']) == 2

    @patch('src.search_engine.search')
    def test_rag_answer_truncates_context(self, mock_search):
        """
        Test rag_answer truncates context to max_context_chars
        Priority: IMPORTANT
        """
        long_text = 'A' * 2000
        mock_search.return_value = [{'title': 'Long Doc', 'text': long_text}]
        
        mock_rag = MagicMock()
        mock_rag.return_value = 'answer'
        
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=Mock(), rag=mock_rag
        )
        
        rag_answer(resources, 'question', max_context_chars=1500)
        
        call_args = mock_rag.call_args
        context_text = call_args[1]['texts'][0]
        assert context_text.startswith('Title: Long Doc\n')
        assert context_text.endswith('A' * 1500)

    @patch('src.search_engine.search')
    def test_rag_answer_passes_domain_filter(self, mock_search):
        """
        Test rag_answer passes domain filter to search
        Priority: CRITICAL
        """
        mock_search.return_value = []
        
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=Mock(), rag=Mock()
        )
        
        rag_answer(resources, 'question', domain='science')
        
        call_args = mock_search.call_args
        assert call_args[1]['domain'] == 'science'


class TestMoreLikeThis:
    """Test suite for more_like_this function"""

    @patch('src.search_engine.search')
    def test_more_like_this_uses_seed_text_as_query(self, mock_search):
        """
        Test more_like_this uses the seed document's text as search query
        Priority: CRITICAL
        """
        mock_search.return_value = []
        
        resources = SearchResources(
            titles=['Title 1', 'Title 2'],
            texts=['Seed text here', 'Other text'],
            domains=['science', 'tech'],
            sources=['source1', 'source2'],
            embeddings=Mock(),
            rag=Mock()
        )
        
        more_like_this(resources, index=0, top_k=5)
        
        # Verify search was called with seed text
        call_args = mock_search.call_args
        assert call_args[0][1] == 'Seed text here'

    @patch('src.search_engine.search')
    def test_more_like_this_filters_seed_document(self, mock_search):
        """
        Test more_like_this filters out the seed document from results
        Priority: CRITICAL
        """
        mock_search.return_value = [
            {'index': 0, 'title': 'Seed'},
            {'index': 1, 'title': 'Similar 1'},
            {'index': 2, 'title': 'Similar 2'}
        ]
        
        resources = SearchResources(
            titles=['Seed', 'Similar 1', 'Similar 2'],
            texts=['text1', 'text2', 'text3'],
            domains=['science'] * 3,
            sources=['source'] * 3,
            embeddings=Mock(),
            rag=Mock()
        )
        
        results = more_like_this(resources, index=0, top_k=5)
        
        # Should exclude the seed document
        assert len(results) == 2
        assert all(r['index'] != 0 for r in results)

    @patch('src.search_engine.search')
    def test_more_like_this_respects_top_k(self, mock_search):
        """
        Test more_like_this returns correct number of results after filtering
        Priority: IMPORTANT
        """
        # Return top_k + 1 to account for seed filtering
        mock_search.return_value = [
            {'index': i, 'title': f'Doc {i}'} for i in range(6)
        ]
        
        resources = SearchResources(
            titles=[f'Doc {i}' for i in range(6)],
            texts=['text'] * 6,
            domains=['science'] * 6,
            sources=['source'] * 6,
            embeddings=Mock(),
            rag=Mock()
        )
        
        results = more_like_this(resources, index=0, top_k=3)
        
        # Should return exactly top_k results after filtering seed
        assert len(results) <= 3


class TestSummarizeResults:
    """Test suite for summarize_results function"""

    @patch('src.search_engine.search')
    @patch('src.search_engine.Summary')
    def test_summarize_results_empty_search(self, mock_summary_class, mock_search):
        """
        Test summarize_results returns empty when no search results
        Priority: IMPORTANT
        """
        mock_search.return_value = []
        
        resources = Mock()
        
        result = summarize_results(resources, 'query')
        
        assert result['summary'] == ''
        assert result['results'] == []

    @patch('src.search_engine.search')
    @patch('src.search_engine.Summary')
    def test_summarize_results_success(self, mock_summary_class, mock_search):
        """
        Test summarize_results combines texts and generates summary
        Priority: NICE-TO-HAVE
        """
        mock_search.return_value = [
            {'text': 'Text 1'},
            {'text': 'Text 2'}
        ]
        
        mock_summarizer = MagicMock()
        mock_summarizer.return_value = 'Summary text'
        mock_summary_class.return_value = mock_summarizer
        
        resources = Mock()
        
        result = summarize_results(resources, 'query', top_k=2)
        
        assert result['summary'] == 'Summary text'
        mock_summarizer.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
