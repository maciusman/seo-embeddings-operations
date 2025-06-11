import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from supabase import Client # For type hinting, actual client passed in
import logging

# Assuming supabase_client.py and its load_embeddings_from_supabase are in utils
try:
    from utils.supabase_client import load_embeddings_from_supabase
except ImportError: # Handle case where utils might not be in python path for direct script run
    logger.info("Could not import load_embeddings_from_supabase from utils.supabase_client for topical_authority_logic.py. This is expected if not running as part of the main app.")
    load_embeddings_from_supabase = None


logger = logging.getLogger(__name__)

def calculate_domain_centroids(
    df_with_domains: pd.DataFrame, # Must contain 'domain' column and index aligned with embeddings_matrix
    embeddings_matrix: np.ndarray,
    streamlit_callback=None
) -> dict[str, np.ndarray]:
    """
    Calculates the centroid for each domain.

    Args:
        df_with_domains: DataFrame containing at least 'url' and 'domain' columns.
                         The index of this DataFrame must correspond to rows in embeddings_matrix.
        embeddings_matrix: NumPy array of embeddings.
        streamlit_callback: Function for UI updates.

    Returns:
        A dictionary mapping domain_name to its centroid_vector.
    """
    domain_centroids = {}
    if df_with_domains.empty or embeddings_matrix.shape[0] != len(df_with_domains):
        logger.error("DataFrame is empty or mismatch between DataFrame length and embeddings matrix rows.")
        if streamlit_callback: streamlit_callback("error","Data or embedding mismatch for centroid calculation.")
        return domain_centroids

    if 'domain' not in df_with_domains.columns:
        logger.error("'domain' column is missing from the DataFrame.")
        if streamlit_callback: streamlit_callback("error", "Domain column missing for centroid calculation.")
        return domain_centroids

    unique_domains = df_with_domains['domain'].dropna().unique() # Ensure NaN domains are skipped
    if streamlit_callback: streamlit_callback("progress_total", len(unique_domains))

    for i, domain in enumerate(unique_domains):
        # Domain already confirmed not to be NaN by dropna().unique()

        if streamlit_callback: streamlit_callback("progress_update", i + 1)
        if streamlit_callback: streamlit_callback("info", f"Calculating centroid for domain: {domain} ({i+1}/{len(unique_domains)})")

        # Correctly get indices from the original DataFrame that match the current domain
        domain_indices = df_with_domains.index[df_with_domains['domain'] == domain].tolist()

        if not domain_indices: # Should not happen if unique_domains is derived from df_with_domains
            logger.warning(f"No pages found for domain '{domain}' (this shouldn't happen). Skipping centroid calculation.")
            continue

        domain_embeddings = embeddings_matrix[domain_indices]

        if domain_embeddings.shape[0] > 0:
            centroid = np.mean(domain_embeddings, axis=0)
            domain_centroids[domain] = centroid
        else:
            logger.warning(f"No embeddings found for domain '{domain}' for specified indices. Skipping centroid.")
            if streamlit_callback: streamlit_callback("warning", f"No embeddings for domain {domain} using selected indices.")

    return domain_centroids

def calculate_site_radius_for_pages(
    df_page_data: pd.DataFrame, # Must contain 'domain' and be indexed consistently with embeddings_matrix
    embeddings_matrix: np.ndarray,
    domain_centroids_map: dict[str, np.ndarray],
    streamlit_callback=None
) -> list[float | None]:
    """
    Calculates the SiteRadius for each page (cosine distance to its own domain's centroid).
    """
    site_radii = [None] * len(df_page_data) # Initialize with Nones
    if 'domain' not in df_page_data.columns:
        logger.error("Missing 'domain' column in page data for SiteRadius calculation.")
        if streamlit_callback: streamlit_callback("error", "Domain column missing for SiteRadius.")
        return site_radii

    if streamlit_callback: streamlit_callback("progress_total", len(df_page_data))

    # Iterate using index to ensure alignment with embeddings_matrix
    for i in range(len(df_page_data)):
        if streamlit_callback: streamlit_callback("progress_update", i + 1)

        row = df_page_data.iloc[i]
        page_domain = row['domain']
        # page_embedding = embeddings_matrix[i:i+1] # Get the i-th row from embeddings_matrix
        # Ensure that the DataFrame index used for domain_indices in centroid calculation
        # is the same as the implicit integer index for embeddings_matrix here.
        # This holds if df_page_data.index is a simple range index [0, 1, ..., n-1]
        # and it was the one used to select domain_indices.
        # The current load_embeddings_from_supabase returns a df with a default range index.

        page_embedding_idx = df_page_data.index[i] # Get the original index if df was filtered/re-indexed
                                                # However, load_embeddings_from_supabase returns a fresh DF
                                                # so iloc[i] for df_page_data corresponds to row i in embeddings_matrix.
        page_embedding = embeddings_matrix[i:i+1]


        if pd.isna(page_domain):
            logger.warning(f"Page {row.get('url', df_page_data.index[i])} has NaN domain. Cannot calculate SiteRadius.")
            # site_radii already initialized with None
            continue

        domain_centroid = domain_centroids_map.get(page_domain)

        if domain_centroid is None:
            logger.warning(f"No centroid found for domain '{page_domain}' (page: {row.get('url', df_page_data.index[i])}). SiteRadius will be None.")
            continue

        distance = cosine_distances(page_embedding, domain_centroid.reshape(1, -1))[0, 0]
        site_radii[i] = float(distance) # Assign to the correct position

    return site_radii


def calculate_topical_authority_metrics(
    supabase_client: Client,
    embedding_type_for_ta: str,
    domain_filter: str | None,
    streamlit_callback
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Main orchestrator for calculating Site Focus and Site Radius metrics.
    """
    if streamlit_callback: streamlit_callback("info", "Starting Topical Authority calculation...")

    df_page_level_data, embeddings_matrix = load_embeddings_from_supabase(
        supabase_client,
        embedding_type_for_ta,
        domain_filter=domain_filter,
        fetch_metadata_cols=['domain', 'title'],
        streamlit_callback=streamlit_callback
    )

    if df_page_level_data is None or embeddings_matrix is None or df_page_level_data.empty:
        if streamlit_callback: streamlit_callback("error", "Failed to load embeddings or no data found.")
        return None, None

    if 'domain' not in df_page_level_data.columns or df_page_level_data['domain'].isnull().all():
        if streamlit_callback: streamlit_callback("error", "No domain information in loaded data.")
        return df_page_level_data, None

    if streamlit_callback: streamlit_callback("info", "Calculating domain centroids...")
    domain_centroids = calculate_domain_centroids(df_page_level_data, embeddings_matrix, streamlit_callback)

    if not domain_centroids:
        if streamlit_callback: streamlit_callback("error", "Failed to calculate domain centroids.")
        return df_page_level_data, None

    if streamlit_callback: streamlit_callback("info", "Calculating SiteRadius for each page...")
    site_radius_values = calculate_site_radius_for_pages(
        df_page_level_data, embeddings_matrix, domain_centroids, streamlit_callback
    )
    df_page_level_data['SiteRadius'] = site_radius_values
    df_page_level_data_cleaned = df_page_level_data.dropna(subset=['SiteRadius']).copy() # Work on a copy after dropna


    if streamlit_callback: streamlit_callback("info", "Aggregating metrics at domain level...")
    if df_page_level_data_cleaned.empty:
        if streamlit_callback: streamlit_callback("warning", "No pages with valid SiteRadius for domain metrics.")
        return df_page_level_data, pd.DataFrame() # Return original page data, empty domain summary

    domain_summary = df_page_level_data_cleaned.groupby('domain').agg(
        AvgSiteRadius=('SiteRadius', 'mean'),
        PageCount=('url', 'count')
    ).reset_index()

    domain_summary['DomainFocus'] = 1 / (1 + domain_summary['AvgSiteRadius'])
    domain_summary = domain_summary.sort_values(by='DomainFocus', ascending=False).reset_index(drop=True)

    if streamlit_callback: streamlit_callback("info", "Topical Authority metrics calculated successfully.")

    return df_page_level_data_cleaned, domain_summary # Return cleaned page data
```
