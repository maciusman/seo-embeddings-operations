import asyncio
import nest_asyncio
import json
import pandas as pd
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup
import logging
from crawl4ai import WebCrawler, PlaywrightCrawler, HttpCrawler
from crawl4ai.utils import get_total_memory_usage, extract_content_from_url, extract_links_from_html, is_crawl_allowed_by_robots
from crawl4ai.utils.content_utils import get_fast_playwright_page, close_playwright_page_and_context
from crawl4ai.utils.filter_utils import DomainFilter, SubPathFilter, RegexFilter, RobotsFilter
from crawl4ai.utils.utils import setup_logging, rate_limit_calls, PlaywrightError, is_url_valid, get_domain
from crawl4ai.bulk_processor import BulkProcessor, BatchProcessor, StreamProcessor

# Apply nest_asyncio to allow running asyncio event loops in environments like Jupyter notebooks or Streamlit
nest_asyncio.apply()

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

def get_content_filter_strategy(bm25_threshold, min_word_threshold, require_h1):
    """
    Creates a content filter strategy instance with the specified parameters.
    """
    from crawl4ai.content_strategy import ContentFilterStrategy
    return ContentFilterStrategy(
        bm25_threshold=bm25_threshold,
        min_word_threshold=min_word_threshold,
        require_h1=require_h1
    )

async def discover_urls(crawler_instance, url_to_crawl, max_depth, max_pages, allowed_url_substring=None):
    """
    Discovers URLs using the crawler instance.
    """
    discovered_urls = await crawler_instance.discover_urls(
        url=url_to_crawl,
        max_depth=max_depth,
        max_pages=max_pages,
        url_filter=SubPathFilter(allowed_url_substring) if allowed_url_substring else None
    )
    return discovered_urls

async def process_single_url(crawler_instance, url, content_verbose):
    """
    Processes a single URL to extract content.
    """
    result = await crawler_instance.process_url(url=url, verbose=content_verbose)
    return result

async def crawl_website_multithreaded(
    url_to_crawl,
    max_depth,
    max_pages,
    allowed_url_substring,
    max_concurrent_threads,
    memory_threshold_percent,
    processing_mode,
    batch_size,
    content_verbose,
    bm25_threshold,
    min_word_threshold,
    require_h1,
    crawler_mode,
    disable_javascript,
    fast_timeouts,
    streamlit_callback=None # For Streamlit progress updates
):
    """
    Main function to crawl a website with multithreading and advanced features.
    Accepts parameters that will be passed from the Streamlit UI.
    """
    if not is_url_valid(url_to_crawl):
        logger.error(f"Invalid URL provided: {url_to_crawl}")
        if streamlit_callback:
            streamlit_callback("error", f"Invalid URL: {url_to_crawl}. Please enter a valid URL (e.g., http://example.com).")
        return pd.DataFrame()

    logger.info(f"Starting crawl for URL: {url_to_crawl}")
    if streamlit_callback:
        streamlit_callback("info", f"Starting crawl for URL: {url_to_crawl}")


    # Choose crawler type
    if crawler_mode.lower() == 'playwright':
        crawler_instance = PlaywrightCrawler(
            max_threads=max_concurrent_threads,
            memory_threshold_percent=memory_threshold_percent,
            disable_javascript=disable_javascript,
            fast_timeouts=fast_timeouts
        )
        logger.info("Using PlaywrightCrawler.")
        if streamlit_callback:
            streamlit_callback("info", "Using PlaywrightCrawler.")
    else:
        crawler_instance = HttpCrawler(
            max_threads=max_concurrent_threads,
            memory_threshold_percent=memory_threshold_percent,
            fast_timeouts=fast_timeouts
        )
        logger.info("Using HttpCrawler.")
        if streamlit_callback:
            streamlit_callback("info", "Using HttpCrawler.")


    # Choose processing mode
    if processing_mode.lower() == 'batch':
        processor = BatchProcessor(crawler_instance, batch_size=batch_size)
    else: # Stream mode
        processor = StreamProcessor(crawler_instance)

    logger.info(f"Processing mode: {processing_mode}")
    if streamlit_callback:
        streamlit_callback("info", f"Processing mode: {processing_mode}")


    # Setup content filter strategy
    content_filter = get_content_filter_strategy(bm25_threshold, min_word_threshold, require_h1)
    processor.set_content_filter_strategy(content_filter)
    logger.info(f"Content filter strategy set with BM25 threshold: {bm25_threshold}, Min word threshold: {min_word_threshold}, Require H1: {require_h1}")
    if streamlit_callback:
        streamlit_callback("info", f"Content filter strategy set with BM25 threshold: {bm25_threshold}, Min word threshold: {min_word_threshold}, Require H1: {require_h1}")

    all_results = []
    crawled_data = []

    try:
        if streamlit_callback:
            streamlit_callback("info", "Discovering URLs...")
        discovered_urls = await discover_urls(crawler_instance, url_to_crawl, max_depth, max_pages, allowed_url_substring)

        if not discovered_urls:
            logger.warning("No URLs discovered. Check the starting URL, max_depth, max_pages, and allowed_url_substring parameters.")
            if streamlit_callback:
                streamlit_callback("warning", "No URLs discovered. Please check your parameters.")
            return pd.DataFrame()

        logger.info(f"Discovered {len(discovered_urls)} URLs. Starting processing...")
        if streamlit_callback:
            streamlit_callback("info", f"Discovered {len(discovered_urls)} URLs. Starting processing...")
            streamlit_callback("progress_total", len(discovered_urls))


        url_count = 0
        async for result in processor.process_urls(discovered_urls, verbose=content_verbose):
            url_count += 1
            if streamlit_callback:
                streamlit_callback("progress_update", url_count)
                streamlit_callback("info", f"Processed {url_count}/{len(discovered_urls)}: {result.url}")

            if result.success:
                all_results.append({
                    "url": result.url,
                    "content": result.content,
                    "metadata": result.metadata,
                    "title": result.metadata.get('title', ''),
                    "h1": result.metadata.get('h1', ''),
                    "status_code": result.status_code,
                    "content_type": result.content_type,
                    "domain": get_domain(result.url),
                    "crawled_at": datetime.now().isoformat()
                })
                logger.debug(f"Successfully processed URL: {result.url} (Status: {result.status_code})")
            else:
                logger.error(f"Failed to process URL: {result.url} (Error: {result.error_message}, Status: {result.status_code})")
                all_results.append({
                    "url": result.url,
                    "content": None,
                    "metadata": None,
                    "title": None,
                    "h1": None,
                    "status_code": result.status_code,
                    "content_type": None,
                    "domain": get_domain(result.url),
                    "crawled_at": datetime.now().isoformat(),
                    "error": result.error_message
                })

        crawled_data = pd.DataFrame(all_results)
        logger.info(f"Crawling complete. Processed {len(all_results)} URLs.")
        if streamlit_callback:
            streamlit_callback("info", f"Crawling complete. Processed {len(all_results)} URLs.")

    except PlaywrightError as e:
        logger.error(f"A Playwright specific error occurred: {e}")
        if streamlit_callback:
            streamlit_callback("error", f"Playwright error: {e}. Ensure Playwright is installed and browsers are set up (try: playwright install).")
    except Exception as e:
        logger.error(f"An unexpected error occurred during crawling: {e}", exc_info=True)
        if streamlit_callback:
            streamlit_callback("error", f"An unexpected error occurred: {e}")
    finally:
        await crawler_instance.close()
        logger.info("Crawler closed.")
        if streamlit_callback:
            streamlit_callback("info", "Crawler resources released.")

    return crawled_data

def start_crawl(params, streamlit_callback):
    """
    Wrapper function to start the crawling process using asyncio.
    `params` is a dictionary containing all crawling parameters from Streamlit UI.
    `streamlit_callback` is a function to send updates to Streamlit.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If an event loop is already running (e.g. in Streamlit context with nest_asyncio)
            # directly await the coroutine.
            future = crawl_website_multithreaded(
                url_to_crawl=params['url_to_crawl'],
                max_depth=params['max_depth'],
                max_pages=params['max_pages'],
                allowed_url_substring=params['allowed_url_substring'],
                max_concurrent_threads=params['max_concurrent_threads'],
                memory_threshold_percent=params['memory_threshold_percent'],
                processing_mode=params['processing_mode'],
                batch_size=params['batch_size'],
                content_verbose=params['content_verbose'],
                bm25_threshold=params['bm25_threshold'],
                min_word_threshold=params['min_word_threshold'],
                require_h1=params['require_h1'],
                crawler_mode=params['crawler_mode'],
                disable_javascript=params['disable_javascript'],
                fast_timeouts=params['fast_timeouts'],
                streamlit_callback=streamlit_callback
            )
            # Create a task and run it if in a context that allows it,
            # otherwise, if nest_asyncio is doing its job, this should work.
            # For Streamlit, it might be better to run this in a separate thread
            # if direct awaiting causes issues, but nest_asyncio should handle it.
            df_results = asyncio.ensure_future(future) # This might need adjustment based on Streamlit's async handling
            # In a typical asyncio setup, you'd await this future.
            # With nest_asyncio in Streamlit, direct call or ensure_future might work.
            # A more robust way for Streamlit might be to run the loop in a separate thread.
            # However, let's try with nest_asyncio's implicit handling first.
            # This part is tricky with Streamlit's execution model.
            # A common pattern is to run the async code using asyncio.run()
            # but that can't be done if a loop is already running.
            # nest_asyncio is supposed to solve this.
            # Let's assume nest_asyncio allows us to call loop.run_until_complete on a new future
            # or that Streamlit's own thread handling + nest_asyncio is sufficient.

            # The most straightforward way with nest_asyncio is often to just await if already in an async context
            # or run if not. Since Streamlit itself isn't async, we rely on nest_asyncio to patch the loop.
            df_results = loop.run_until_complete(future)

        else:
            df_results = asyncio.run(crawl_website_multithreaded(
                url_to_crawl=params['url_to_crawl'],
                max_depth=params['max_depth'],
                max_pages=params['max_pages'],
                allowed_url_substring=params['allowed_url_substring'],
                max_concurrent_threads=params['max_concurrent_threads'],
                memory_threshold_percent=params['memory_threshold_percent'],
                processing_mode=params['processing_mode'],
                batch_size=params['batch_size'],
                content_verbose=params['content_verbose'],
                bm25_threshold=params['bm25_threshold'],
                min_word_threshold=params['min_word_threshold'],
                require_h1=params['require_h1'],
                crawler_mode=params['crawler_mode'],
                disable_javascript=params['disable_javascript'],
                fast_timeouts=params['fast_timeouts'],
                streamlit_callback=streamlit_callback
            ))
        return df_results
    except RuntimeError as e:
        if "cannot run event loop while another loop is running" in str(e) and streamlit_callback:
            streamlit_callback("error", "Asyncio loop conflict. This is an issue with running async code in Streamlit. Try restarting the app.")
            # This indicates nest_asyncio might not be perfectly handling the Streamlit case,
            # or there's a deeper conflict.
        elif streamlit_callback:
            streamlit_callback("error", f"Runtime error during crawl setup: {e}")
        logger.error(f"Runtime error starting crawl: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        if streamlit_callback:
            streamlit_callback("error", f"Unexpected error starting crawl: {e}")
        logger.error(f"Unexpected error starting crawl: {e}", exc_info=True)
        return pd.DataFrame()

if __name__ == '__main__':
    # This is a placeholder for testing the crawler logic independently
    # In a real scenario, this would be called from app.py

    # Example parameters (replace with actual values or UI inputs)
    test_params = {
        'url_to_crawl': "https://www.example.com",
        'max_depth': 1,
        'max_pages': 5,
        'allowed_url_substring': "", # No specific substring filter
        'max_concurrent_threads': 5,
        'memory_threshold_percent': 90,
        'processing_mode': "Stream", # Stream or Batch
        'batch_size': 10, # Relevant for Batch mode
        'content_verbose': False,
        'bm25_threshold': 0.3,
        'min_word_threshold': 50,
        'require_h1': False,
        'crawler_mode': "HTTP", # HTTP or Playwright
        'disable_javascript': False, # Relevant for Playwright
        'fast_timeouts': True
    }

    def dummy_streamlit_callback(type, message):
        print(f"[Streamlit Dummy Callback] {type.upper()}: {message}")
        if type == "progress_total":
            # Initialize a dummy progress bar
            global pbar_total
            pbar_total = message # Store total for dummy tqdm
            print(f"Total URLs to process: {pbar_total}")
        if type == "progress_update":
            # Update dummy progress bar
            print(f"Progress: {message}/{pbar_total}")


    print("Simulating a crawl from __main__ (for testing crawler_logic.py)...")
    # Since this is an async function, we need to run it in an event loop
    # results_df = asyncio.run(crawl_website_multithreaded(**test_params, streamlit_callback=dummy_streamlit_callback))

    # Using the new start_crawl wrapper
    results_df = start_crawl(test_params, dummy_streamlit_callback)

    if not results_df.empty:
        print("\nCrawling Results:")
        print(results_df.head())
        # Save to CSV
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"crawled_data_{timestamp}.csv"
        # results_df.to_csv(filename, index=False)
        # print(f"\nResults saved to {filename}")
    else:
        print("\nNo results obtained from crawling.")
