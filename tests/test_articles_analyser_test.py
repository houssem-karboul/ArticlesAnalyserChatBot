import unittest
from unittest.mock import patch, MagicMock
from main import process_urls, answer_query  # Import the functions from main.py
import pickle


"""
1/ The tests utilize mocking to isolate the functionality of the functions being tested 
   and to avoid external dependencies.
2/ Ensure that the test environment has unittest and any other necessary packages installed.
"""

class TestArticlesAnalyserChatBot(unittest.TestCase):
    """
    Test suite for the Articles Analyser ChatBot application.
    
    This class contains unit tests for the main functionalities of the ChatBot, 
    specifically the URL processing and query answering functionalities.
    """

    @patch('main.UnstructuredURLLoader')
    @patch('main.RecursiveCharacterTextSplitter')
    @patch('main.NVIDIAEmbeddings')
    @patch('main.FAISS')
    @patch('main.pickle.dump')
    @patch('main.main_placeholder.text')
    def test_process_urls_success(self, mock_text, mock_dump, mock_faiss, mock_embeddings, mock_text_splitter, mock_loader):
        """
        Test the process_urls function for successful URL processing.
        This test simulates the loading of articles from given URLs, 
        checks if the text splitter and embeddings are invoked correctly, 
        and verifies that the FAISS index is saved.
        """
        
        # Mock class to simulate document structure
        class MockDoc:
            def __init__(self, content, source):
                self.content = content
                self.metadata = {'source': source}

        # Arrange: Set up mock return values for the dependencies
        mock_loader.return_value.load.return_value = [
            {'content': 'Article 1', 'url': 'http://example.com/1'},
            {'content': 'Article 2', 'url': 'http://example.com/2'}
        ]
        
        # Ensure that split_documents returns objects with a 'metadata' attribute
        mock_text_splitter.return_value.split_documents.return_value = [
            MockDoc('Chunk 1', 'http://example.com/1'),
            MockDoc('Chunk 2', 'http://example.com/2')
        ]

        mock_embeddings.return_value = MagicMock()
        mock_vectorstore = MagicMock()
        mock_faiss.from_documents.return_value = mock_vectorstore

        # Act: Call the function under test
        process_urls(['http://example.com/1', 'http://example.com/2'])

        # Assert: Verify that the expected methods were called
        mock_text.assert_any_call("Data Loading...Started...✅✅✅")
        mock_text.assert_any_call("Text Splitter...Started...✅✅✅")
        mock_text.assert_any_call("Embedding Vector Started Building...✅✅✅")
        mock_dump.assert_called_once()  # Ensure the FAISS index was saved
        mock_faiss.from_documents.assert_called_once()  # Ensure embeddings were created


        @patch('main.UnstructuredURLLoader')
        def test_process_urls_invalid_url(self, mock_loader):
            """
            Test process_urls function with an invalid URL.
            
            This test ensures that the function raises an exception when an invalid URL is processed.
            """
            # Arrange: Set up the mock to raise an exception
            mock_loader.side_effect = Exception("Invalid URL")
            
            # Act & Assert: Check that the exception is raised
            with self.assertRaises(Exception) as context:
                process_urls(['invalid_url'])
            self.assertEqual(str(context.exception), "Invalid URL")

        @patch('main.pickle.load')
        @patch('main.FAISS')
        @patch('main.RetrievalQAWithSourcesChain.from_llm')
        @patch('main.main_placeholder.text_input')
        def test_answer_query_success(self, mock_text_input, mock_chain, mock_faiss_load):
            """
            Test the answer_query function for a successful query response.
            
            This test simulates a successful answer retrieval from the ChatBot 
            and checks if the answer and sources are correctly processed.
            """
            # Arrange: Set up mock return values
            mock_faiss_load.return_value = MagicMock()
            mock_chain.return_value = {
                'answer': 'This is a test answer', 
                'sources': 'http://example.com/1\nhttp://example.com/2'
            }
            mock_text_input.return_value = 'What is the answer?'
            
            # Act: Call the function under test
            answer_query('What is the answer?')
            
            # Assert: Verify that the answer and sources were processed
            self.assertIn("Answer", mock_chain.call_args_list[0][0][0])  # Check if answer was processed
            self.assertIn("Sources:", mock_chain.call_args_list[0][0][0])  # Check if sources were processed

        @patch('main.pickle.load')
        def test_answer_query_no_faiss_store(self, mock_faiss_load):
            """
            Test answer_query function when the FAISS store does not exist.
            
            This test ensures that a FileNotFoundError is raised when the FAISS store
            is not available.
            """
            # Arrange: Set up the mock to raise a FileNotFoundError
            mock_faiss_load.side_effect = FileNotFoundError
            
            # Act & Assert: Check that the exception is raised
            with self.assertRaises(FileNotFoundError):
                answer_query('What is the answer?')

        if __name__ == '__main__':
            unittest.main()