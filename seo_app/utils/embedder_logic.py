import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, RetryError, TooManyRequests
import time
import pandas as pd
import tiktoken # For token counting
import logging
from utils.supabase_client import get_supabase_client, get_page_by_url, insert_crawled_page, get_embeddings_for_page, insert_page_embedding
from urllib.parse import urlparse # Added for domain extraction if not present
from datetime import datetime # Added for crawl_timestamp if not present


logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Configure if running standalone

# --- Constants ---
# Default model, can be overridden by UI
DEFAULT_EMBEDDING_MODEL = "models/embedding-001"
# Max tokens for Gemini embedding models (e.g., embedding-001 is 2048 for text, 1 image)
# This is for the input text. The output dimension is fixed (e.g., 768).
# For "models/text-embedding-004", the max input tokens is 2048.
# For "models/embedding-001", it's also 2048.
DEFAULT_MAX_TOKENS_INPUT = 2048

# --- Tiktoken Encoder ---
# Used for robustly counting tokens to stay within model limits.
# "cl100k_base" is the encoding used by text-embedding-ada-002 and newer OpenAI models.
# For Google's models, the exact tokenizer isn't specified as being identical to OpenAI's,
# but cl100k_base is a common general-purpose tokenizer.
# Google's own examples often use `len(text.split())` or similar, which is less accurate.
# Using tiktoken provides a more standard approach to estimate token count.
# If Google provides a specific tokenizer for their models, that should be used.
# For now, cl100k_base is a reasonable proxy.
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    logger.warning("Tiktoken cl100k_base tokenizer not found, falling back to splitting by space for token counting.")
    tokenizer = None

def count_tokens(text: str) -> int:
    """Counts tokens in a string using tiktoken if available, else approximates."""
    if not text:
        return 0
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text.split()) # Basic approximation

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncates text to a maximum number of tokens."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text) # Ensure text is string

    if tokenizer:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            return tokenizer.decode(truncated_tokens)
        return text
    else: # Fallback if tiktoken is not available
        words = text.split()
        if len(words) > max_tokens: # Approximate with words if no tokenizer
            return " ".join(words[:max_tokens])
        return text

def prepare_metadata_text(row: pd.Series, max_tokens: int) -> str:
    """
    Prepares a concise text string from metadata for embedding.
    Includes title, meta description, and H1 headings.
    Truncates the combined text to max_tokens.
    """
    title = row.get('title', '') or ''
    meta_description = row.get('meta_description', '') or ''
    h1 = row.get('h1', '') or row.get('h1_headings', '') or '' # 'h1' from Colab, 'h1_headings' from crawler_logic

    # Ensure all parts are strings
    title = str(title) if pd.notna(title) else ''
    meta_description = str(meta_description) if pd.notna(meta_description) else ''
    h1 = str(h1) if pd.notna(h1) else ''

    # Combine metadata with clear separators
    combined_text = f"Title: {title}\nMeta Description: {meta_description}\nH1: {h1}"

    # Remove extra whitespace
    combined_text = ' '.join(combined_text.split())

    return truncate_text(combined_text, max_tokens)

def generate_embedding_google(
    text_to_embed: str,
    api_key: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    task_type: str = "RETRIEVAL_DOCUMENT", # Default task type for document embeddings
    title: str = None, # Optional title for RETRIEVAL_DOCUMENT task type
    streamlit_callback=None
) -> tuple[list[float] | None, str, int | None]:
    """
    Generates an embedding for the given text using Google's Generative AI.

    Args:
        text_to_embed: The text to embed.
        api_key: The Google API key.
        model_name: The name of the embedding model to use.
        task_type: The task type for the embedding.
        title: Optional title if task_type is 'RETRIEVAL_DOCUMENT'.
        streamlit_callback: Function to send updates/errors to Streamlit.

    Returns:
        A tuple containing:
        - list[float]: The embedding vector, or None if an error occurred.
        - str: Status message ('success', 'error: <reason>').
        - int: Number of tokens in the input text, or None.
    """
    if not text_to_embed or not text_to_embed.strip():
        logger.warning("Attempted to embed empty or whitespace-only text.")
        if streamlit_callback:
            streamlit_callback("warning", "Skipping empty text for embedding.")
        return None, "error: empty text", 0

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to configure Google AI SDK with API key: {e}", exc_info=True)
        if streamlit_callback:
            streamlit_callback("error", f"Google API Key configuration error: {e}")
        return None, f"error: API key configuration - {e}", None

    token_count = count_tokens(text_to_embed)

    try:
        logger.debug(f"Generating embedding with model {model_name} for text (first 50 chars): '{text_to_embed[:50]}...'")

        # Prepare arguments for embed_content
        embed_args = {
            "model": model_name,
            "content": text_to_embed,
            "task_type": task_type,
        }
        if task_type == "RETRIEVAL_DOCUMENT" and title:
            embed_args["title"] = title

        embedding_response = genai.embed_content(**embed_args)
        embedding_vector = embedding_response.get('embedding')

        if not embedding_vector:
            logger.error(f"Embedding generation failed. Response: {embedding_response}")
            error_msg = f"error: API returned no embedding. Response: {embedding_response}"
            if streamlit_callback:
                streamlit_callback("error", error_msg)
            return None, error_msg, token_count

        logger.debug(f"Successfully generated embedding. Vector dimension: {len(embedding_vector)}")
        if streamlit_callback: # Be cautious with too many success messages for individual embeddings
             pass # streamlit_callback("info", f"Embedding generated for: {text_to_embed[:30]}...")
        return embedding_vector, "success", token_count

    except TooManyRequests as e:
        logger.warning(f"Rate limit hit for Google API: {e}. Consider increasing MIN_INTERVAL or reducing requests_per_minute.")
        error_msg = f"error: rate limit exceeded - {e}"
        if streamlit_callback:
            streamlit_callback("error", error_msg)
        return None, error_msg, token_count
    except GoogleAPIError as e: # Catch more general Google API errors
        logger.error(f"Google API error during embedding generation: {e}", exc_info=True)
        error_msg = f"error: Google API - {e}"
        if streamlit_callback:
            streamlit_callback("error", error_msg)
        return None, error_msg, token_count
    except RetryError as e: # Catch retry errors
        logger.error(f"RetryError during embedding generation: {e}", exc_info=True)
        error_msg = f"error: Retry failed - {e}"
        if streamlit_callback:
            streamlit_callback("error", error_msg)
        return None, error_msg, token_count
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
        error_msg = f"error: unexpected - {e}"
        if streamlit_callback:
            streamlit_callback("error", error_msg)
        return None, error_msg, token_count


def generate_embeddings_for_df(
    df: pd.DataFrame,
    text_column_name: str, # Column in df to get text from (e.g., 'main_content_raw' or a prepared metadata column)
    embedding_model_name: str,
    task_type: str,
    max_tokens_input: int,
    api_key: str,
    requests_per_minute: int,
    embedding_type_name: str, # e.g., 'metadata_gemini_004', 'content_gemini_004'
    regenerate_embeddings: bool = False, # Flag to control re-generation
    streamlit_callback=None,
    supabase_client=None # Pass Supabase client for DB operations
):
    """
    Generates embeddings for a specified text column in a DataFrame and stores them in Supabase.

    Args:
        df: Pandas DataFrame containing the text data.
        text_column_name: The name of the column with text to embed.
                          If 'metadata', special preparation is done.
        embedding_model_name: Name of the Google embedding model.
        task_type: Task type for embedding generation.
        max_tokens_input: Max tokens for input text truncation.
        api_key: Google Gemini API Key.
        requests_per_minute: To control API call rate.
        embedding_type_name: A descriptive name for this embedding type (e.g., 'metadata_gemini_004').
        regenerate_embeddings: If True, will regenerate even if embedding exists.
        streamlit_callback: For UI updates.
        supabase_client: Initialized Supabase client.
    """
    if df.empty:
        if streamlit_callback: streamlit_callback("warning", "Input DataFrame is empty. Nothing to embed.")
        return 0, 0 # Processed, Succeeded

    if not supabase_client:
        if streamlit_callback: streamlit_callback("error", "Supabase client not provided. Cannot store embeddings.")
        return 0, 0

    min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.1 # Min interval between API calls

    processed_count = 0
    success_count = 0

    if streamlit_callback:
        streamlit_callback("progress_total", len(df)) # For progress bar

    for index, row in df.iterrows():
        processed_count += 1
        if streamlit_callback:
            streamlit_callback("progress_update", processed_count)
            streamlit_callback("info", f"Processing row {processed_count}/{len(df)}: {row.get('url', 'N/A')}")

        page_url = row.get('url')
        if not page_url:
            if streamlit_callback: streamlit_callback("warning", f"Skipping row {index} due to missing URL.")
            continue

        # 1. Get or Create Page ID in Supabase
        page_data_for_db = {
            'url': page_url,
            'domain': row.get('domain', urlparse(page_url).netloc if page_url else None),
            'crawl_timestamp': row.get('crawled_at', datetime.now().isoformat()), # Use 'crawled_at' from crawler
            'h1_headings': row.get('h1', row.get('h1_headings')), # Allow both 'h1' and 'h1_headings'
            'title': row.get('title'),
            'meta_description': row.get('meta_description'),
            'main_content_raw': row.get('content', row.get('main_content_raw')), # 'content' from crawler, 'main_content_raw' for general
            'keywords': row.get('keywords'),
            'status_code': row.get('status_code'),
            'content_type': row.get('content_type'),
            'crawled_by': row.get('crawler_mode', 'unknown') # e.g. HTTP or Playwright
        }
        # Remove None values before insert to avoid issues with Supabase types
        page_data_for_db_cleaned = {k: v for k, v in page_data_for_db.items() if v is not None}


        page_id_in_db = None
        try:
            # Check if page exists first to get ID, then insert if not.
            # This is slightly less efficient than direct upsert for new pages but gives more control.
            existing_page = get_page_by_url(supabase_client, page_url)
            if existing_page:
                page_id_in_db = existing_page['id']
                # Optionally, update crawled_pages if new data is fresher (not implemented here for simplicity)
                if streamlit_callback: streamlit_callback("info", f"Page {page_url} exists in DB with ID {page_id_in_db}.")
            else:
                page_id_in_db = insert_crawled_page(supabase_client, page_data_for_db_cleaned)
                if page_id_in_db:
                    if streamlit_callback: streamlit_callback("info", f"Page {page_url} inserted into DB with ID {page_id_in_db}.")
                else:
                    if streamlit_callback: streamlit_callback("error", f"Failed to insert page {page_url} into DB.")
                    continue # Skip to next row if page cannot be saved
        except Exception as e:
            if streamlit_callback: streamlit_callback("error", f"DB error processing page {page_url}: {e}")
            logger.error(f"DB error for page {page_url}: {e}", exc_info=True)
            continue


        if not page_id_in_db:
            if streamlit_callback: streamlit_callback("error", f"Could not obtain page_id for {page_url}. Skipping embedding.")
            continue

        # 2. Prepare text for embedding
        if text_column_name == "metadata":
            text_to_embed = prepare_metadata_text(row, max_tokens_input)
            # Use page title for RETRIEVAL_DOCUMENT task type if available
            embed_title = row.get('title')
        elif text_column_name in df.columns:
            raw_text = str(row[text_column_name]) if pd.notna(row[text_column_name]) else ""
            text_to_embed = truncate_text(raw_text, max_tokens_input)
            embed_title = row.get('title') # Can also be used for content embeddings
        else:
            if streamlit_callback: streamlit_callback("error", f"Text column '{text_column_name}' not found in DataFrame for URL {page_url}.")
            continue

        if not text_to_embed.strip():
            if streamlit_callback: streamlit_callback("warning", f"Skipping URL {page_url} as there is no text to embed for column '{text_column_name}'.")
            continue

        # 3. Check if embedding already exists (if not regenerating)
        if not regenerate_embeddings:
            existing_embedding = get_embeddings_for_page(supabase_client, page_id_in_db, embedding_type_name, embedding_model_name, task_type)
            if existing_embedding:
                if streamlit_callback: streamlit_callback("info", f"Embedding for {page_url} ({embedding_type_name}, {embedding_model_name}) already exists. Skipping.")
                success_count +=1 # Count as success if already exists and not regenerating
                continue

        # 4. Generate Embedding
        time.sleep(min_interval) # Respect rate limits

        embedding_vector, status, token_count = generate_embedding_google(
            text_to_embed,
            api_key,
            embedding_model_name,
            task_type,
            title=embed_title, # Pass title if available and task type is RETRIEVAL_DOCUMENT
            streamlit_callback=streamlit_callback
        )

        # 5. Store Embedding in Supabase
        if embedding_vector and status == "success":
            emb_id = insert_page_embedding(
                supabase_client, page_id_in_db, embedding_type_name,
                embedding_vector, embedding_model_name, task_type, token_count
            )
            if emb_id:
                success_count += 1
                if streamlit_callback: streamlit_callback("info", f"Successfully embedded and stored for {page_url} ({embedding_type_name}).")
            else:
                if streamlit_callback: streamlit_callback("error", f"Failed to store embedding for {page_url} ({embedding_type_name}).")
        else:
            if streamlit_callback: streamlit_callback("error", f"Failed to generate embedding for {page_url} ({embedding_type_name}): {status}")

    if streamlit_callback:
        streamlit_callback("info", f"Embedding generation complete. Processed: {processed_count}, Succeeded: {success_count}.")
    return processed_count, success_count


if __name__ == '__main__':
    # This is a placeholder for testing the embedder logic independently
    # Requires a running Supabase instance and a valid Google API Key.

    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Dummy Streamlit Callback for testing ---
    def dummy_streamlit_callback(type, message):
        print(f"[DUMMY STREAMLIT] {type.upper()}: {message}")
        if type == "progress_total":
            global pbar_total_dummy
            pbar_total_dummy = message
        if type == "progress_update":
            print(f"Progress: {message}/{pbar_total_dummy}")

    # --- Test Data and Parameters (Replace with your actual data) ---
    # TEST_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY" # Replace with your key
    # TEST_SUPABASE_URL = "YOUR_SUPABASE_URL"     # Replace
    # TEST_SUPABASE_KEY = "YOUR_SUPABASE_KEY"     # Replace

    # if "YOUR_GOOGLE_GEMINI_API_KEY" in TEST_API_KEY or \
    #    "YOUR_SUPABASE_URL" in TEST_SUPABASE_URL or \
    #    "YOUR_SUPABASE_KEY" in TEST_SUPABASE_KEY:
    #     logger.error("Please set your API Key and Supabase credentials in the __main__ block for testing.")
    # else:
    #     # Initialize Supabase client (make sure supabase_client.py is in the same directory or installed)
    #     from utils.supabase_client import init_supabase_client, ensure_tables_exist
    #     supabase_client_instance = init_supabase_client(TEST_SUPABASE_URL, TEST_SUPABASE_KEY)

    #     if supabase_client_instance and ensure_tables_exist(supabase_client_instance):
    #         logger.info("Supabase client initialized and tables ensured for testing.")

    #         # Sample DataFrame (mimicking crawled data)
    #         data = {
    #             'url': ['http://example.com/page1', 'http://example.com/page2', 'http://example.com/page3-empty'],
    #             'title': ['Test Page 1', 'Another Test Page 2', 'Empty Page'],
    #             'meta_description': ['Description for page 1.', 'Details about page 2.', ''],
    #             'h1_headings': ['Header One of Page 1', 'Page 2 Main H1', 'No H1'],
    #             'main_content_raw': [
    #                 "This is the full main content of the first page. It has several sentences and provides useful information.",
    #                 "Page two discusses other topics. It is also very informative and well-written, aiming to engage the reader.",
    #                 "" # Empty content for testing
    #             ],
    #             'domain': ['example.com'] * 3,
    #             'crawled_at': [datetime.now().isoformat()] * 3,
    #             'status_code': [200] * 3,
    #             'content_type': ['text/html'] * 3,
    #             'crawler_mode': ['HTTP'] * 3
    #         }
    #         sample_df = pd.DataFrame(data)

    #         logger.info(f"Test DataFrame created with {len(sample_df)} rows.")

    #         # --- Test Metadata Embeddings ---
    #         logger.info("--- Starting Metadata Embedding Test ---")
    #         processed_meta, success_meta = generate_embeddings_for_df(
    #             df=sample_df.copy(), # Use a copy to avoid modifications if any
    #             text_column_name="metadata", # Special keyword for metadata preparation
    #             embedding_model_name="models/embedding-001", # Or "models/text-embedding-004"
    #             task_type="RETRIEVAL_DOCUMENT", # Suitable for metadata used in retrieval
    #             max_tokens_input=1024, # Max tokens for combined metadata
    #             api_key=TEST_API_KEY,
    #             requests_per_minute=20, # Adjust based on your API limits
    #             embedding_type_name="metadata_gemini_test_001", # Descriptive name for DB
    #             regenerate_embeddings=True, # Set to False to test skipping existing
    #             streamlit_callback=dummy_streamlit_callback,
    #             supabase_client=supabase_client_instance
    #         )
    #         logger.info(f"Metadata Embedding Test Complete. Processed: {processed_meta}, Succeeded: {success_meta}")

    #         # --- Test Content Embeddings ---
    #         logger.info("--- Starting Content Embedding Test ---")
    #         processed_content, success_content = generate_embeddings_for_df(
    #             df=sample_df.copy(),
    #             text_column_name="main_content_raw", # Actual column name for content
    #             embedding_model_name="models/embedding-001",
    #             task_type="RETRIEVAL_DOCUMENT", # Or "SEMANTIC_SIMILARITY" depending on use case
    #             max_tokens_input=2048, # Max tokens for content
    #             api_key=TEST_API_KEY,
    #             requests_per_minute=20,
    #             embedding_type_name="content_gemini_test_001",
    #             regenerate_embeddings=True,
    #             streamlit_callback=dummy_streamlit_callback,
    #             supabase_client=supabase_client_instance
    #         )
    #         logger.info(f"Content Embedding Test Complete. Processed: {processed_content}, Succeeded: {success_content}")

    #     else:
    #         logger.error("Failed to initialize Supabase or ensure tables for testing.")

    logger.info("Embedder logic defined. For testing, provide API key, Supabase creds, and uncomment __main__.")
    # from datetime import datetime # for main_content_raw in generate_embeddings_for_df # Already imported
    # from urllib.parse import urlparse # for domain in generate_embeddings_for_df # Already imported
    # Added imports for datetime and urlparse that were missed in the main body of the function but used in the example/test section.
    # Corrected 'h1' vs 'h1_headings' and 'content' vs 'main_content_raw' access in page_data_for_db.
    # Added token_count to generate_embedding_google return and insert_page_embedding call.
    # Added title parameter to generate_embedding_google and used it in generate_embeddings_for_df.
    # Made sure all parts of metadata text are strings before combining.
    # Ensured text_to_embed is not empty before API call.
    # Used tiktoken for more accurate token counting and truncation.
    # Added error handling for genai.configure.
    # Added more specific exception handling for Google API errors.
    # Refined the logic for getting/creating page_id in Supabase.
    # Added missing imports of urlparse and datetime at the top level of the module.
```
