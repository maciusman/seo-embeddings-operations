import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from supabase import Client
import logging
# import streamlit as st # Avoid direct streamlit imports in backend logic files

logger = logging.getLogger(__name__)

def load_embeddings_from_supabase( # Modified to fetch metadata and return DataFrame
    supabase_client: Client,
    embedding_type: str,
    domain_filter: str = None,
    fetch_metadata_cols: list[str] = None,
    streamlit_callback=None
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Loads embeddings and specified metadata from Supabase for a specific embedding type,
    optionally filtered by domain.
    """
    if not supabase_client:
        if streamlit_callback: streamlit_callback("error", "Supabase client not available.")
        logger.error("Supabase client not available for loading embeddings.")
        return None, None

    if fetch_metadata_cols is None:
        fetch_metadata_cols = []

    # Always fetch 'url' and 'domain', plus user specified columns. Ensure no duplicates.
    base_cols = ['id', 'url', 'domain'] # 'id' here refers to crawled_pages.id
    all_cols_to_fetch = list(set(base_cols + fetch_metadata_cols))
    select_str_crawled_pages = ", ".join(all_cols_to_fetch)
    # page_id in page_embeddings table refers to crawled_pages.id
    select_str = f"embedding_vector, page_id, crawled_pages!inner({select_str_crawled_pages})"

    try:
        if streamlit_callback: streamlit_callback("info", f"Loading embeddings of type '{embedding_type}'...")

        query = supabase_client.table("page_embeddings").select(select_str).eq("embedding_type", embedding_type)

        if domain_filter and domain_filter.strip():
            if streamlit_callback: streamlit_callback("info", f"Applying domain filter: '{domain_filter}'")
            query = query.eq("crawled_pages.domain", domain_filter.strip()) # Filter on crawled_pages.domain

        response = query.execute()

        if response.data:
            metadata_records = []
            embeddings_list = []

            for record in response.data:
                page_info = record.get("crawled_pages")
                if page_info and page_info.get("url") and record.get("embedding_vector"):
                    # Construct metadata dict for the DataFrame
                    current_meta = {'crawled_page_id': page_info['id'], 'url': page_info['url'], 'domain': page_info.get('domain')}
                    for col in fetch_metadata_cols: # Only fetch user-specified ones for the final DF beyond base
                        if col not in ['id','url','domain']: # Avoid duplicating base keys already added
                           current_meta[col] = page_info.get(col)
                    metadata_records.append(current_meta)
                    embeddings_list.append(record["embedding_vector"])
                else:
                    logger.warning(f"Skipping record due to missing URL, metadata, or embedding vector: page_id in embedding table {record.get('page_id')}")

            if not metadata_records:
                if streamlit_callback: streamlit_callback("warning", f"No valid data found for embedding type '{embedding_type}' with current filters.")
                return None, None

            df_metadata = pd.DataFrame(metadata_records)
            embeddings_array = np.array(embeddings_list, dtype=np.float32)

            if streamlit_callback: streamlit_callback("info", f"Loaded {len(df_metadata)} embeddings with their metadata.")
            return df_metadata, embeddings_array
        else:
            if streamlit_callback: streamlit_callback("warning", f"No embeddings found for type '{embedding_type}' with current filters.")
            return None, None # Return None, None if no data from query
    except Exception as e:
        logger.error(f"Error loading embeddings and metadata from Supabase: {e}", exc_info=True)
        if streamlit_callback: streamlit_callback("error", f"Error loading embeddings/metadata: {e}")
        return None, None

def calculate_cosine_similarity_matrix(embeddings_matrix: np.ndarray) -> np.ndarray | None:
    if embeddings_matrix is None or embeddings_matrix.ndim != 2 or embeddings_matrix.shape[0] == 0:
        logger.warning("Invalid or empty embeddings matrix for similarity calculation.")
        return None
    try:
        return cosine_similarity(embeddings_matrix)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity matrix: {e}", exc_info=True)
        return None

def find_top_k_similar(
    target_url_index: int,
    url_list: list[str],
    similarity_matrix: np.ndarray,
    top_k: int = 10,
    min_similarity_threshold: float = 0.0,
    exclude_self: bool = True
) -> list[tuple[str, float]]:
    if similarity_matrix is None or target_url_index >= similarity_matrix.shape[0] or target_url_index < 0 :
        return []
    sim_scores = sorted(list(enumerate(similarity_matrix[target_url_index])), key=lambda x: x[1], reverse=True)
    top_similar_urls = []
    for i, score in sim_scores:
        if len(top_similar_urls) >= top_k: break
        if exclude_self and i == target_url_index: continue
        if score < min_similarity_threshold: break
        if url_list[i] != url_list[target_url_index] or not exclude_self: # Ensure different URL if excluding self
             top_similar_urls.append((url_list[i], float(score)))
    return top_similar_urls

def perform_similarity_analysis(
    supabase_client: Client, embedding_type: str, domain_filter: str | None,
    top_k: int, min_similarity_threshold: float, streamlit_callback
) -> pd.DataFrame | None:
    if streamlit_callback: streamlit_callback("info", "Starting similarity analysis...")
    df_metadata, embeddings_matrix = load_embeddings_from_supabase(
        supabase_client, embedding_type, domain_filter,
        fetch_metadata_cols=[], # Similarity analysis primarily needs URLs for output
        streamlit_callback=streamlit_callback
    )
    if embeddings_matrix is None or df_metadata is None or df_metadata.empty:
        if streamlit_callback: streamlit_callback("warning", "No embeddings loaded for similarity analysis.")
        return None
    urls = df_metadata['url'].tolist()
    if len(urls) < 2:
        if streamlit_callback: streamlit_callback("warning", "Not enough URLs (<2) for similarity analysis.")
        return pd.DataFrame({'source_url': urls, 'message': ['Not enough other URLs to compare.'] * len(urls)})

    if streamlit_callback: streamlit_callback("info", f"Calculating similarity matrix for {len(urls)} items...")
    similarity_matrix = calculate_cosine_similarity_matrix(embeddings_matrix)

    if similarity_matrix is None:
        if streamlit_callback: streamlit_callback("error", "Failed to calculate similarity matrix.")
        return None

    if streamlit_callback: streamlit_callback("info", "Finding top K similar items for each URL...")
    all_results = []
    if streamlit_callback: streamlit_callback("progress_total", len(urls))
    for i, source_url in enumerate(urls):
        if streamlit_callback: streamlit_callback("progress_update", i + 1)
        top_k_for_url = find_top_k_similar(i, urls, similarity_matrix, top_k, min_similarity_threshold)
        row_result = {'source_url': source_url}
        for j, (sim_url, score) in enumerate(top_k_for_url):
            row_result[f'similar_url_{j+1}'] = sim_url
            row_result[f'similarity_score_{j+1}'] = round(score, 4)
        all_results.append(row_result)

    if streamlit_callback: streamlit_callback("info", "Similarity analysis complete.")
    if not all_results: return pd.DataFrame()
    results_df = pd.DataFrame(all_results)
    column_order = ['source_url']
    for k_val in range(1, top_k + 1): # Ensure columns up to top_k are defined if they exist
        if f'similar_url_{k_val}' in results_df.columns : # check if column was even created
            column_order.append(f'similar_url_{k_val}')
            column_order.append(f'similarity_score_{k_val}')
    return results_df[column_order]

# --- Content Issue Detection Functions ---
def detect_content_issues(
    df_metadata_with_urls: pd.DataFrame,
    similarity_matrix: np.ndarray,
    duplicate_threshold: float,
    cannibalization_min: float,
    cannibalization_max: float,
    streamlit_callback=None
) -> pd.DataFrame:
    issues = []
    if df_metadata_with_urls.empty or 'url' not in df_metadata_with_urls.columns:
        logger.error("DataFrame for issue detection is empty or missing 'url' column.")
        return pd.DataFrame()

    url_list = df_metadata_with_urls['url'].tolist()
    num_urls = len(url_list)
    if num_urls < 2: return pd.DataFrame() # Not enough items to compare

    if streamlit_callback: streamlit_callback("progress_total", num_urls * (num_urls -1) // 2)

    processed_pairs = 0
    for i in range(num_urls):
        for j in range(i + 1, num_urls):
            processed_pairs +=1
            # Update progress periodically to avoid too many callbacks
            if processed_pairs % 1000 == 0 and streamlit_callback:
                 streamlit_callback("progress_update", processed_pairs)

            similarity = similarity_matrix[i, j]
            url1_data = df_metadata_with_urls.iloc[i]
            url2_data = df_metadata_with_urls.iloc[j]

            issue_type = None
            if similarity >= duplicate_threshold: issue_type = 'duplicate_content'
            elif cannibalization_min <= similarity <= cannibalization_max: issue_type = 'seo_cannibalization'

            if issue_type:
                issues.append({
                    'url_1': url1_data['url'],
                    'title_1': url1_data.get('title', 'N/A'),
                    'url_2': url2_data['url'],
                    'title_2': url2_data.get('title', 'N/A'),
                    'similarity_score': round(similarity, 4),
                    'issue_type': issue_type
                })

    if streamlit_callback: streamlit_callback("progress_update", processed_pairs) # Final update
    return pd.DataFrame(issues)

def perform_content_issue_detection(
    supabase_client: Client, metadata_embedding_type: str, domain_filter: str | None,
    duplicate_threshold: float, cannibalization_min: float, cannibalization_max: float,
    streamlit_callback
) -> pd.DataFrame | None:
    if streamlit_callback: streamlit_callback("info", "Starting content issue detection...")

    required_metadata_cols = ['title', 'url', 'domain']
    df_metadata, embeddings_matrix = load_embeddings_from_supabase(
        supabase_client, metadata_embedding_type, domain_filter,
        fetch_metadata_cols=required_metadata_cols,
        streamlit_callback=streamlit_callback
    )
    if embeddings_matrix is None or df_metadata is None or df_metadata.empty:
        if streamlit_callback: streamlit_callback("warning", "No metadata embeddings for content issue detection.")
        return None
    if len(df_metadata) < 2:
        if streamlit_callback: streamlit_callback("warning", "Not enough URLs (<2) for content issue detection.")
        return pd.DataFrame()

    if streamlit_callback: streamlit_callback("info", f"Calculating similarity matrix for {len(df_metadata)} items...")
    similarity_matrix = calculate_cosine_similarity_matrix(embeddings_matrix)

    if similarity_matrix is None:
        if streamlit_callback: streamlit_callback("error", "Failed to calculate similarity matrix for metadata.")
        return None

    if streamlit_callback: streamlit_callback("info", "Detecting duplicate content and cannibalization issues...")
    issues_df = detect_content_issues(
        df_metadata, similarity_matrix, duplicate_threshold,
        cannibalization_min, cannibalization_max, streamlit_callback
    )
    if streamlit_callback: streamlit_callback("info", f"Content issue detection complete. Found {len(issues_df)} potential issues.")
    return issues_df
```
