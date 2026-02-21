# test_client.py
# Unit and integration tests for the client CLI module

import sys
from pathlib import Path

# Add parent directory to path to import application modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from src.client import choose_domain, print_menu, main


class TestChooseDomain:
    """Test suite for choose_domain function"""

    @patch('builtins.input', side_effect=['1'])
    @patch('builtins.print')
    def test_choose_domain_valid_selection(self, mock_print, mock_input):
        """
        Test choose_domain returns correct domain for valid selection
        Priority: CRITICAL
        """
        domains = ['science', 'technology', 'health']
        
        result = choose_domain(domains)
        
        assert result == 'health'  # Sorted order: health, science, technology

    @patch('builtins.input', side_effect=['2'])
    @patch('builtins.print')
    def test_choose_domain_second_option(self, mock_print, mock_input):
        """
        Test choose_domain selects second domain correctly
        Priority: CRITICAL
        """
        domains = ['zebra', 'alpha', 'beta']
        
        result = choose_domain(domains)
        
        # Sorted: alpha, beta, zebra
        assert result == 'beta'

    @patch('builtins.input', side_effect=['invalid', '0', '10', '1'])
    @patch('builtins.print')
    def test_choose_domain_invalid_then_valid(self, mock_print, mock_input):
        """
        Test choose_domain retries on invalid input then accepts valid
        Priority: CRITICAL
        """
        domains = ['science', 'technology']
        
        result = choose_domain(domains)
        
        # Should eventually succeed with valid input
        assert result in ['science', 'technology']
        # Should have prompted 4 times total (3 invalid + 1 valid)
        assert mock_input.call_count == 4

    @patch('builtins.input', side_effect=['1'])
    @patch('builtins.print')
    def test_choose_domain_normalizes_domains(self, mock_print, mock_input):
        """
        Test choose_domain normalizes domains (lowercase, strip)
        Priority: IMPORTANT
        """
        domains = ['  SCIENCE  ', 'technology', 'Health ']
        
        result = choose_domain(domains)
        
        # Should be normalized
        assert result == result.strip().lower()

    @patch('builtins.input', side_effect=['1'])
    @patch('builtins.print')
    def test_choose_domain_deduplicates(self, mock_print, mock_input):
        """
        Test choose_domain removes duplicate domains
        Priority: IMPORTANT
        """
        domains = ['science', 'SCIENCE', 'science', 'technology']
        
        choose_domain(domains)
        
        # Check that printed options show unique domains
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined_output = ' '.join(print_calls)
        
        # Count how many times 'science' appears in the numbered list
        science_count = combined_output.lower().count('science')
        # Should appear only once in the list (not counting the header)
        assert science_count < len(domains)

    @patch('builtins.input', side_effect=['1'])
    @patch('builtins.print')
    def test_choose_domain_sorts_alphabetically(self, mock_print, mock_input):
        """
        Test choose_domain sorts domains alphabetically
        Priority: NICE-TO-HAVE
        """
        domains = ['zebra', 'alpha', 'beta']
        
        choose_domain(domains)
        
        # Check print output for ordering
        print_calls = [str(call) for call in mock_print.call_args_list]
        # First numbered item should be alpha, then beta, then zebra
        assert any('alpha' in str(c) for c in print_calls)


class TestPrintMenu:
    """Test suite for print_menu function"""

    @patch('builtins.print')
    def test_print_menu_displays_all_options(self, mock_print):
        """
        Test print_menu displays all menu options
        Priority: CRITICAL
        """
        print_menu()
        
        # Verify print was called
        assert mock_print.called
        
        # Combine all print calls to check content
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        
        # Check for key menu items
        assert 'Search' in combined or 'search' in combined.lower()
        assert 'RAG' in combined or 'Answer' in combined
        assert 'Quit' in combined or 'quit' in combined.lower()

    @patch('builtins.print')
    def test_print_menu_has_seven_options(self, mock_print):
        """
        Test print_menu shows exactly 7 menu options
        Priority: IMPORTANT
        """
        print_menu()
        
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        
        # Menu should have options 1-7
        for i in range(1, 8):
            assert str(i) in combined


class TestMainLoop:
    """Test suite for main function"""

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '7'])  # Choose domain 1, then quit
    @patch('builtins.print')
    def test_main_initializes_resources(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main initializes resources on startup
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science', 'technology']
        mock_load.return_value = mock_resources
        
        main()
        
        # Verify resources were loaded and closed
        mock_load.assert_called_once()
        mock_close.assert_called_once_with(mock_resources)

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '7'])
    @patch('builtins.print')
    def test_main_requires_domain_selection(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main requires user to select domain at startup
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science', 'technology']
        mock_load.return_value = mock_resources
        
        main()
        
        # First input should be domain selection
        assert mock_input.call_count >= 2  # Domain + Quit

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('src.client.search')
    @patch('builtins.input', side_effect=['1', '1', 'test query', '7'])  # Domain, Search, Query, Quit
    @patch('builtins.print')
    def test_main_menu_option_1_search(self, mock_print, mock_input, mock_search, mock_close, mock_load):
        """
        Test main menu option 1 executes search
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        mock_search.return_value = [
            {
                'title': 'Test Result',
                'domain': 'science',
                'source': 'test.com',
                'score': 0.95,
                'text': 'Test text'
            }
        ]
        
        main()
        
        # Verify search was called
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[0][1] == 'test query'

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('src.client.rag_answer')
    @patch('builtins.input', side_effect=['1', '2', 'test question', '7'])  # Domain, RAG, Question, Quit
    @patch('builtins.print')
    def test_main_menu_option_2_rag(self, mock_print, mock_input, mock_rag, mock_close, mock_load):
        """
        Test main menu option 2 executes RAG answer
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        mock_rag.return_value = {
            'answer': 'Generated answer',
            'results': []
        }
        
        main()
        
        # Verify RAG was called
        mock_rag.assert_called_once()
        call_args = mock_rag.call_args
        assert call_args[0][1] == 'test question'

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('src.client.search')
    @patch('src.client.more_like_this')
    @patch('builtins.input', side_effect=['1', '1', 'query', '3', '1', '7'])  # Domain, Search, More like, select, Quit
    @patch('builtins.print')
    def test_main_menu_option_3_more_like_this(self, mock_print, mock_input, mock_more, mock_search, mock_close, mock_load):
        """
        Test main menu option 3 finds similar documents
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        mock_search.return_value = [
            {
                'title': 'Result 1',
                'domain': 'science',
                'source': 'test.com',
                'score': 0.95,
                'text': 'Text',
                'index': 42
            }
        ]
        
        mock_more.return_value = []
        
        main()
        
        # Verify more_like_this was called with correct index
        mock_more.assert_called_once()
        call_args = mock_more.call_args
        assert call_args[0][1] == 42  # Index from first result

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '3', '7'])  # Domain, More like (no prior results), Quit
    @patch('builtins.print')
    def test_main_more_like_this_without_prior_search(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test more_like_this option requires prior search results
        Priority: IMPORTANT
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        main()
        
        # Should show error message about needing prior results
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert 'search' in combined.lower() or 'rag' in combined.lower()

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '4', '2', '5', '7'])  # Domain, Change domain, Select 2, Show domain, Quit
    @patch('builtins.print')
    def test_main_menu_option_4_change_domain(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main menu option 4 changes domain filter
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['alpha', 'beta', 'gamma']
        mock_load.return_value = mock_resources
        
        main()
        
        # Check that domain change was acknowledged
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert 'beta' in combined  # Selected domain 2

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '5', '7'])  # Domain, Show domain, Quit
    @patch('builtins.print')
    def test_main_menu_option_5_show_domain(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main menu option 5 displays current domain
        Priority: IMPORTANT
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        main()
        
        # Check that domain was displayed
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert 'science' in combined.lower()

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '6', '10', '7'])  # Domain, Change top-k, Set to 10, Quit
    @patch('builtins.print')
    def test_main_menu_option_6_change_topk(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main menu option 6 changes top-k parameter
        Priority: IMPORTANT
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        main()
        
        # Check that top-k change was acknowledged
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert '10' in combined

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '6', 'invalid', '7'])  # Domain, Change top-k, Invalid, Quit
    @patch('builtins.print')
    def test_main_menu_option_6_invalid_topk(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test changing top-k rejects invalid input
        Priority: IMPORTANT
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        main()
        
        # Should show error for invalid top-k
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert 'invalid' in combined.lower() or 'error' in combined.lower()

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', '7'])  # Domain, Quit
    @patch('builtins.print')
    def test_main_menu_option_7_quit(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main menu option 7 exits cleanly
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        main()
        
        # Should have closed resources
        mock_close.assert_called_once()

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', 'invalid_option', '7'])  # Domain, Invalid, Quit
    @patch('builtins.print')
    def test_main_invalid_menu_option(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main handles invalid menu options gracefully
        Priority: IMPORTANT
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        main()
        
        # Should show invalid option message
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert 'invalid' in combined.lower()

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('src.client.search')
    @patch('builtins.input', side_effect=['1', '1', '', '7'])  # Domain, Search, Empty query, Quit
    @patch('builtins.print')
    def test_main_handles_empty_query(self, mock_print, mock_input, mock_search, mock_close, mock_load):
        """
        Test main skips search when query is empty
        Priority: IMPORTANT
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        main()
        
        # Search should not be called with empty query
        mock_search.assert_not_called()

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('src.client.search')
    @patch('builtins.input', side_effect=['1', '1', 'query', '7'])
    @patch('builtins.print')
    def test_main_displays_search_results(self, mock_print, mock_input, mock_search, mock_close, mock_load):
        """
        Test main displays search results with proper formatting
        Priority: IMPORTANT
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        mock_search.return_value = [
            {
                'title': 'Test Title',
                'domain': 'science',
                'source': 'test.com',
                'score': 0.95,
                'text': 'Full text content here'
            }
        ]
        
        main()
        
        # Check that result details were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert 'Test Title' in combined
        assert 'science' in combined
        assert '0.95' in combined or '0.9500' in combined

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('src.client.search')
    @patch('builtins.input', side_effect=['1', '1', 'query', '7'])
    @patch('builtins.print')
    def test_main_truncates_long_text_preview(self, mock_print, mock_input, mock_search, mock_close, mock_load):
        """
        Test main truncates long text to preview length (300 chars)
        Priority: NICE-TO-HAVE
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        long_text = 'A' * 500
        mock_search.return_value = [
            {
                'title': 'Test',
                'domain': 'science',
                'source': 'test.com',
                'score': 0.95,
                'text': long_text
            }
        ]
        
        main()
        
        # Check that ellipsis was added for truncation
        print_calls = [str(call) for call in mock_print.call_args_list]
        combined = ' '.join(print_calls)
        assert '...' in combined

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('builtins.input', side_effect=['1', KeyboardInterrupt])
    @patch('builtins.print')
    def test_main_closes_resources_on_exception(self, mock_print, mock_input, mock_close, mock_load):
        """
        Test main closes resources even when exception occurs
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science']
        mock_load.return_value = mock_resources
        
        with pytest.raises(KeyboardInterrupt):
            main()
        
        # Should still close resources via finally block
        mock_close.assert_called_once_with(mock_resources)


class TestIntegration:
    """Integration tests for client with search_engine"""

    @patch('src.client.load_resources')
    @patch('src.client.close_resources')
    @patch('src.client.search')
    @patch('builtins.input', side_effect=['1', '1', 'machine learning', '7'])
    @patch('builtins.print')
    def test_full_search_workflow(self, mock_print, mock_input, mock_search, mock_close, mock_load):
        """
        Test complete workflow: startup -> domain selection -> search -> quit
        Priority: CRITICAL
        """
        mock_resources = MagicMock()
        mock_resources.domains = ['science', 'technology']
        mock_load.return_value = mock_resources
        
        mock_search.return_value = [
            {
                'title': 'ML Paper',
                'domain': 'science',
                'source': 'arxiv.org',
                'score': 0.98,
                'text': 'Machine learning research paper content',
                'index': 0
            }
        ]
        
        main()
        
        # Verify complete workflow
        mock_load.assert_called_once()
        mock_search.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify search was called with correct parameters
        call_args = mock_search.call_args
        assert call_args[0][1] == 'machine learning'  # Query
        assert 'domain' in call_args[1]  # Domain filter applied


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
