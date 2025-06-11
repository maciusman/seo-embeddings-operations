import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler # StandardScaler currently commented out
import google.generativeai as genai # For LLM-based naming
import time
import logging

# Assuming supabase_client.py and its load_embeddings_from_supabase are in utils
# This will be adjusted in app.py to be from .supabase_client import ...
# For standalone testing of this module, ensure supabase_client.py is accessible.
try:
    from utils.supabase_client import load_embeddings_from_supabase
except ImportError: # Handle case where utils might not be in python path for direct script run
    logger.info("Could not import load_embeddings_from_supabase from utils.supabase_client for clustering_logic.py. This is expected if not running as part of the main app.")
    load_embeddings_from_supabase = None


logger = logging.getLogger(__name__)

# --- Helper: Tiktoken for LLM Naming (if needed for truncation/prompts) ---
try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    logger.warning("Tiktoken cl100k_base tokenizer not found, falling back to splitting by space for token counting if needed for LLM prompts.")
    tokenizer = None

def count_tokens_for_llm(text: str) -> int:
    if not text: return 0
    if tokenizer: return len(tokenizer.encode(text))
    return len(text.split())

def truncate_text_for_llm(text: str, max_tokens: int) -> str:
    if not text: return ""
    if not isinstance(text, str): text = str(text)
    if tokenizer:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            return tokenizer.decode(tokens[:max_tokens])
        return text
    else:
        words = text.split()
        if len(words) > max_tokens: return " ".join(words[:max_tokens])
        return text

# --- Clustering Core Functions ---

def determine_optimal_clusters_silhouette(
    embeddings_matrix: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
    streamlit_callback=None
) -> int:
    """
    Determines the optimal number of clusters using Silhouette Score.
    """
    if embeddings_matrix is None :
        logger.error("Embeddings matrix is None. Cannot determine optimal clusters.")
        if streamlit_callback: streamlit_callback("error", "Embeddings data missing for Silhouette analysis.")
        return min_k # Return min_k as a fallback

    if embeddings_matrix.shape[0] < min_k :
        logger.warning(f"Not enough samples ({embeddings_matrix.shape[0]}) for silhouette analysis with min_k={min_k}. Defaulting to {embeddings_matrix.shape[0] if embeddings_matrix.shape[0] > 0 else 1 } clusters.")
        if streamlit_callback: streamlit_callback("warning", f"Too few samples for robust cluster number determination. Using {embeddings_matrix.shape[0] if embeddings_matrix.shape[0] > 0 else 1} clusters.")
        return embeddings_matrix.shape[0] if embeddings_matrix.shape[0] > 0 else 1


    scaled_embeddings = embeddings_matrix # No scaling for now, as mentioned in original Colab

    best_k = min_k
    highest_silhouette_score = -1.01 # Silhouette scores range from -1 to 1

    if streamlit_callback: streamlit_callback("info", f"Determining optimal clusters (k={min_k} to {max_k}) using Silhouette Score...")

    actual_max_k = min(max_k, scaled_embeddings.shape[0] -1 if scaled_embeddings.shape[0] > 1 else 2)
    # Ensure min_k is not greater than actual_max_k if actual_max_k became too small
    actual_min_k = min(min_k, actual_max_k)
    if actual_min_k <=1 and actual_max_k >=2 : actual_min_k =2 # kmeans needs at least 2 clusters for silhouette
    elif actual_min_k <=1 : # if max_k is also 1 or less (e.g. 1 sample)
        logger.warning(f"Sample size too small ({scaled_embeddings.shape[0]}) for range {min_k}-{max_k}. Setting k=1.")
        if streamlit_callback: streamlit_callback("warning", f"Sample size too small for range {min_k}-{max_k}. Setting k=1.")
        return 1


    for k_val in range(actual_min_k, actual_max_k + 1):
        try:
            if streamlit_callback: streamlit_callback("info", f"Testing k={k_val}...")
            kmeans = KMeans(n_clusters=k_val, random_state=random_state, n_init='auto', init='k-means++')
            cluster_labels = kmeans.fit_predict(scaled_embeddings)

            if len(set(cluster_labels)) < 2:
                logger.warning(f"Only one distinct cluster found for k={k_val}. Silhouette score not applicable or will be -1.")
                current_silhouette_score = -1.0
            else:
                current_silhouette_score = silhouette_score(scaled_embeddings, cluster_labels)

            logger.info(f"For k={k_val}, Silhouette Score: {current_silhouette_score:.4f}")

            if current_silhouette_score > highest_silhouette_score:
                highest_silhouette_score = current_silhouette_score
                best_k = k_val
        except Exception as e:
            logger.error(f"Error during silhouette analysis for k={k_val}: {e}", exc_info=True)
            if streamlit_callback: streamlit_callback("error", f"Error finding optimal k for {k_val}: {e}")

    logger.info(f"Optimal k determined: {best_k} with Silhouette Score: {highest_silhouette_score:.4f}")
    if streamlit_callback: streamlit_callback("info", f"Optimal number of clusters (k) found: {best_k} (Silhouette Score: {highest_silhouette_score:.3f})")
    return best_k if best_k >= actual_min_k else actual_min_k # Ensure it returns at least min_k

def perform_flat_clustering(
    embeddings_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> np.ndarray | None:
    if embeddings_matrix is None:
        logger.error("Embeddings matrix is None for flat clustering.")
        return None
    if n_clusters <= 0:
        logger.error(f"n_clusters ({n_clusters}) must be > 0 for flat clustering.")
        return None # Or raise error

    if embeddings_matrix.shape[0] < n_clusters:
        logger.warning(f"Number of samples ({embeddings_matrix.shape[0]}) is less than n_clusters ({n_clusters}). Adjusting n_clusters to {embeddings_matrix.shape[0]}.")
        n_clusters = embeddings_matrix.shape[0]
        if n_clusters == 0: return np.array([])

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto', init='k-means++')
        cluster_labels = kmeans.fit_predict(embeddings_matrix)
        logger.info(f"Flat clustering performed. Assigned {len(set(cluster_labels))} unique cluster labels for {n_clusters} requested clusters.")
        return cluster_labels
    except Exception as e:
        logger.error(f"Error during flat clustering: {e}", exc_info=True)
        return None

def get_cluster_centroids(embeddings_matrix: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centroids = {}
    if embeddings_matrix is None or labels is None: return centroids
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1: continue
        cluster_points = embeddings_matrix[labels == label]
        if cluster_points.shape[0] > 0:
            centroids[label] = np.mean(cluster_points, axis=0)
    return centroids

def perform_hierarchical_clustering(
    df_embeddings: pd.DataFrame,
    embeddings_matrix: np.ndarray,
    num_super_clusters: int,
    sub_cluster_range_min: int = 2,
    sub_cluster_range_max: int = 5,
    min_samples_for_subcluster: int = 10,
    random_state: int = 42,
    streamlit_callback=None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if embeddings_matrix is None or num_super_clusters <= 0:
        logger.error("Embeddings matrix is None or num_super_clusters is invalid for hierarchical clustering.")
        return None, None

    n_samples = embeddings_matrix.shape[0]
    if n_samples == 0: return np.array([]), np.array([])
    if n_samples < num_super_clusters:
        logger.warning(f"Adjusting num_super_clusters from {num_super_clusters} to {n_samples} due to insufficient samples.")
        num_super_clusters = n_samples

    if streamlit_callback: streamlit_callback("info", f"Performing Level 1 clustering into {num_super_clusters} super-clusters...")
    super_cluster_labels = perform_flat_clustering(embeddings_matrix, num_super_clusters, random_state)

    if super_cluster_labels is None:
        if streamlit_callback: streamlit_callback("error", "Failed to perform super-clustering.")
        return None, None

    sub_cluster_labels_final = np.full(n_samples, -1, dtype=int)
    unique_super_labels = np.unique(super_cluster_labels)
    if streamlit_callback: streamlit_callback("progress_total", len(unique_super_labels))

    for i, super_label in enumerate(unique_super_labels):
        if streamlit_callback:
            streamlit_callback("progress_update", i + 1)
            streamlit_callback("info", f"Processing Super-Cluster {super_label + 1}/{len(unique_super_labels)}...")

        super_cluster_indices = np.where(super_cluster_labels == super_label)[0]

        if len(super_cluster_indices) < min_samples_for_subcluster or len(super_cluster_indices) < sub_cluster_range_min :
            if streamlit_callback: streamlit_callback("info", f"Super-Cluster {super_label+1} too small ({len(super_cluster_indices)} items) for sub-clustering. Assigning all to sub-cluster 0.")
            sub_cluster_labels_final[super_cluster_indices] = 0 # Assign to a default sub-cluster 0
            continue

        super_cluster_embeddings = embeddings_matrix[super_cluster_indices]
        if streamlit_callback: streamlit_callback("info", f"Determining optimal sub-clusters for Super-Cluster {super_label+1} ({len(super_cluster_indices)} items)...")

        current_max_sub_k = min(sub_cluster_range_max, super_cluster_embeddings.shape[0] -1 if super_cluster_embeddings.shape[0] > 1 else 2)
        current_min_sub_k = sub_cluster_range_min
        if current_max_sub_k < current_min_sub_k : current_max_sub_k = current_min_sub_k # Ensure max >= min
        if current_min_sub_k <=1 and current_max_sub_k >=2 : current_min_sub_k =2
        elif current_min_sub_k <=1: # if max_k is also 1 or less
             if streamlit_callback: streamlit_callback("info", f"Super-Cluster {super_label+1} too small for further sub-division after range adjustment. Assigning to sub-cluster 0.")
             sub_cluster_labels_final[super_cluster_indices] = 0
             continue

        num_sub_clusters = determine_optimal_clusters_silhouette(
            super_cluster_embeddings, current_min_sub_k, current_max_sub_k, random_state, streamlit_callback
        )

        if num_sub_clusters < 2 :
             if streamlit_callback: streamlit_callback("info", f"Optimal sub-clusters for Super-Cluster {super_label+1} is {num_sub_clusters}. Assigning to sub-cluster 0.")
             sub_cluster_labels_final[super_cluster_indices] = 0
             continue

        if streamlit_callback: streamlit_callback("info", f"Sub-clustering Super-Cluster {super_label+1} into {num_sub_clusters} sub-clusters...")
        sub_labels_for_super_cluster = perform_flat_clustering(super_cluster_embeddings, num_sub_clusters, random_state)

        if sub_labels_for_super_cluster is not None:
            sub_cluster_labels_final[super_cluster_indices] = sub_labels_for_super_cluster
        else: # Failed to sub-cluster
            if streamlit_callback: streamlit_callback("error", f"Failed to sub-cluster Super-Cluster {super_label+1}. Assigning to sub-cluster 0.")
            sub_cluster_labels_final[super_cluster_indices] = 0 # Fallback

    logger.info("Hierarchical clustering performed.")
    return super_cluster_labels, sub_cluster_labels_final

def generate_cluster_names_llm(
    df_cluster_subset: pd.DataFrame, text_for_naming_col: str,
    llm_model_name: str, max_name_words: int, api_key: str,
    max_texts_for_prompt: int = 10, max_tokens_per_text_sample: int = 100,
    streamlit_callback=None
) -> str:
    if df_cluster_subset.empty or text_for_naming_col not in df_cluster_subset.columns:
        logger.warning("DataFrame subset is empty or text column for naming is missing.")
        return "Unnamed Cluster (No Data)"
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"LLM Naming: Google API Key config error: {e}", exc_info=True)
        if streamlit_callback: streamlit_callback("error", f"LLM Naming: API Key config error - {e}")
        return "Unnamed Cluster (API Key Error)"

    sample_texts = [
        truncate_text_for_llm(str(row[text_for_naming_col]), max_tokens_per_text_sample)
        for _, row in df_cluster_subset.sample(n=min(max_texts_for_prompt, len(df_cluster_subset)), random_state=42).iterrows()
        if str(row.get(text_for_naming_col, "")).strip()
    ]
    if not sample_texts: return "Unnamed Cluster (No Text Samples)"
    combined_samples = "\n- ".join(sample_texts)
    prompt = f"Analyze these text samples from a content cluster:\n- {combined_samples}\n\nGenerate a concise name ({max_name_words} words max) for this cluster, capturing its primary theme. Cluster Name:"

    try:
        # if streamlit_callback: streamlit_callback("info", f"Generating name with LLM ({llm_model_name})...") # Can be too noisy
        model = genai.GenerativeModel(llm_model_name)
        response = model.generate_content(prompt)
        cluster_name = response.text.strip().replace('"', '').replace("'", "")
        if len(cluster_name.split()) > max_name_words:
            cluster_name = " ".join(cluster_name.split()[:max_name_words]) + "..."
        logger.info(f"Generated cluster name: '{cluster_name}'")
        return cluster_name if cluster_name else "Unnamed Cluster (LLM Error)"
    except Exception as e:
        logger.error(f"Error during LLM cluster naming with {llm_model_name}: {e}", exc_info=True)
        if streamlit_callback: streamlit_callback("error", f"LLM Naming Error ({llm_model_name}): {e}")
        return "Unnamed Cluster (LLM Exception)"

def run_clustering_analysis(
    supabase_client, embedding_type_for_clustering: str, text_for_llm_col: str,
    domain_filter: str | None, clustering_mode: str,
    auto_determine_clusters: bool = True, manual_clusters: int = 5,
    cluster_range_min: int = 2, cluster_range_max: int = 10,
    num_super_clusters: int = 3, hier_sub_cluster_range_min: int = 2,
    hier_sub_cluster_range_max: int = 5, min_samples_for_subcluster_hier: int = 10,
    generate_names_llm: bool = False, llm_naming_model: str | None = None,
    max_cluster_name_words: int = 5, gemini_api_key: str | None = None,
    random_state: int = 42, streamlit_callback=None
) -> pd.DataFrame | None:
    if streamlit_callback: streamlit_callback("info", "Starting Clustering Analysis...")

    metadata_cols_to_fetch = [text_for_llm_col, 'title', 'url', 'domain'] # Ensure necessary cols are fetched
    df_data, embeddings_matrix = load_embeddings_from_supabase(
        supabase_client, embedding_type_for_clustering, domain_filter,
        fetch_metadata_cols=list(set(metadata_cols_to_fetch)),
        streamlit_callback=streamlit_callback
    )
    if df_data is None or embeddings_matrix is None or df_data.empty:
        if streamlit_callback: streamlit_callback("error", "Failed to load embeddings or no data found.")
        return None
    if text_for_llm_col not in df_data.columns and generate_names_llm:
         if streamlit_callback: streamlit_callback("error", f"Column '{text_for_llm_col}' for LLM naming not in data. LLM Naming will be skipped.")
         generate_names_llm = False

    if clustering_mode == 'Flat':
        n_clusters = manual_clusters
        if auto_determine_clusters:
            n_clusters = determine_optimal_clusters_silhouette(
                embeddings_matrix, cluster_range_min, cluster_range_max, random_state, streamlit_callback
            )
        if streamlit_callback: streamlit_callback("info", f"Performing Flat Clustering into {n_clusters} clusters...")
        labels = perform_flat_clustering(embeddings_matrix, n_clusters, random_state)
        if labels is not None: df_data['cluster_id'] = labels.astype(str)
        else:
            if streamlit_callback: streamlit_callback("error", "Flat clustering failed.")
            return df_data
    elif clustering_mode == 'Hierarchical':
        if streamlit_callback: streamlit_callback("info", "Performing Hierarchical Clustering...")
        super_labels, sub_labels = perform_hierarchical_clustering(
            df_data, embeddings_matrix, num_super_clusters,
            hier_sub_cluster_range_min, hier_sub_cluster_range_max,
            min_samples_for_subcluster_hier, random_state, streamlit_callback
        )
        if super_labels is not None:
            df_data['super_cluster_id'] = super_labels
            df_data['sub_cluster_id'] = sub_labels
            df_data['cluster_id'] = [f"{s}_{sub}" if sub != -1 else str(s) for s, sub in zip(super_labels, sub_labels)]
        else:
            if streamlit_callback: streamlit_callback("error", "Hierarchical clustering failed.")
            return df_data
    else: # Should not happen with UI selectbox
        if streamlit_callback: streamlit_callback("error", f"Unknown clustering mode: {clustering_mode}")
        return df_data

    if generate_names_llm and 'cluster_id' in df_data.columns and llm_naming_model and gemini_api_key:
        if streamlit_callback: streamlit_callback("info", "Generating cluster names using LLM...")
        cluster_names_map = {}
        unique_cluster_ids = df_data['cluster_id'].unique()
        if streamlit_callback: streamlit_callback("progress_total", len(unique_cluster_ids))
        for i, cid_val in enumerate(unique_cluster_ids):
            if streamlit_callback:
                streamlit_callback("progress_update", i + 1)
                streamlit_callback("info", f"Naming cluster: {cid_val} ({i+1}/{len(unique_cluster_ids)})")
            cluster_subset = df_data[df_data['cluster_id'] == cid_val]
            if not cluster_subset.empty:
                name = generate_cluster_names_llm(
                    cluster_subset, text_for_llm_col, llm_naming_model,
                    max_cluster_name_words, gemini_api_key, streamlit_callback=None
                )
                cluster_names_map[cid_val] = name
            else: cluster_names_map[cid_val] = f"Cluster {cid_val} (empty)"
        df_data['cluster_name'] = df_data['cluster_id'].map(cluster_names_map)
        if streamlit_callback: streamlit_callback("info", "Cluster naming complete.")
    elif generate_names_llm:
        if streamlit_callback: streamlit_callback("warning", "LLM Naming skipped (check API key, model, or if clusters were generated).")

    if streamlit_callback: streamlit_callback("info", "Clustering analysis finished.")
    return df_data
```
