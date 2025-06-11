# SEO Embeddings & Analysis Web Application

## 1. Introduction

This web application provides a suite of tools for advanced SEO analysis, leveraging the power of content embeddings, SupaBase for data storage, and Google's Generative AI for tasks like embedding generation and content naming. It migrates and expands upon functionalities typically found in data science notebooks (like Jupyter or Colab) into a user-friendly Streamlit interface.

The primary purpose of this application is to empower SEO professionals and content strategists with data-driven insights for tasks such as content auditing, identifying topical gaps, detecting content duplication or cannibalization, understanding site structure through clustering, and assessing topical authority.

## 2. Features

The application offers the following key functionalities:

*   **Web Crawling:**
    *   Utilizes `crawl4ai` for efficient website crawling.
    *   Supports both HTTP and Playwright (JavaScript rendering) modes.
    *   Advanced filtering options including BM25-based content relevance filtering.
    *   Configuration for crawl depth, max pages, concurrency, and more.
*   **Embedding Generation:**
    *   Generates embeddings for both metadata (titles, descriptions, H1s) and main content.
    *   Uses Google Generative AI models (e.g., `models/embedding-001`, `models/text-embedding-004`).
    *   User-configurable task types for embeddings (e.g., `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`).
*   **SupaBase Integration:**
    *   Stores crawled page data (`crawled_pages` table) and their corresponding embeddings (`page_embeddings` table) in a SupaBase PostgreSQL database.
    *   Ensures `pgvector` extension is available and attempts to create tables if they don't exist.
    *   Retrieves data and embeddings from SupaBase for various analysis tasks.
*   **Content Similarity Analysis:**
    *   Calculates and displays the most similar pages to each URL based on their embedding similarity.
    *   Configurable Top-K and similarity threshold.
*   **Duplicate Content & SEO Cannibalization Detection:**
    *   Uses metadata embedding similarity to identify potential duplicate content or SEO cannibalization issues.
    *   User-defined thresholds for classifying issues.
    *   Provides actionable insights based on similarity scores.
*   **Content Clustering:**
    *   Performs flat (KMeans) or two-level hierarchical clustering on content or metadata embeddings.
    *   Option for automatic determination of optimal cluster numbers using Silhouette Score.
    *   LLM-based cluster naming using Google Generative AI to provide human-readable labels for each cluster.
*   **Topical Authority (Site Focus & Site Radius):**
    *   Calculates domain centroids based on content embeddings.
    *   Determines `SiteRadius` for each page (distance to its own domain's centroid).
    *   Aggregates metrics to calculate `AvgSiteRadius` and `DomainFocus` for each domain.
*   **Interactive Visualizations:**
    *   PCA and t-SNE scatter plots to visualize content clusters (using Plotly).
    *   Histograms for SiteRadius distribution within domains.
    *   Scatter plot for comparing DomainFocus vs. AvgSiteRadius.
*   **Data Download:**
    *   Allows users to download results from various analyses (crawled data, similarity reports, content issues, clustering results, topical authority metrics) as CSV files.

## 3. Prerequisites

*   **Python:** Version 3.10+ (specifically, `python-3.11.7` is specified in `runtime.txt`).
*   **SupaBase Project:** Access to an active SupaBase project. You'll need its URL and Anon Key.
*   **Google Gemini API Key:** Required for generating embeddings and for LLM-based cluster naming.

## 4. Setup and Installation (Local)

Follow these steps to set up and run the application on your local machine:

### Step 4.1: Clone the Repository
```bash
git clone <repository_url>
cd seo_app
```
*(Replace `<repository_url>` with the actual URL of the Git repository.)*

### Step 4.2: Create a Virtual Environment (Recommended)
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### Step 4.3: Install Dependencies
Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
If using the Playwright crawler mode, you may also need to install browser binaries if `crawl4ai-setup` (or a similar command) wasn't run or did not complete successfully:
```bash
playwright install --with-deps chromium
# or just `playwright install` to get default browsers
```
(Often, `crawl4ai` itself handles Playwright setup during its first use or via a setup command.)

### Step 4.4: SupaBase Setup
1.  Go to [SupaBase](https://supabase.com/) and sign up or log in.
2.  Create a new project or use an existing one.
3.  Navigate to the **SQL Editor** in your SupaBase project dashboard.
4.  Ensure the `vector` extension (required for storing embeddings) is enabled by running:
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```
5.  Go to **Project Settings > API** to find your **Project URL** and **Anon Key** (public). You will need these for the application.
6.  **Important Note on DDL for tables:** The application, when connecting to SupaBase, will attempt to create the necessary tables (`crawled_pages`, `page_embeddings`) if they don't already exist, using the schema defined in `utils/supabase_client.py`.
    *   However, enabling extensions like `vector` or creating tables might require specific permissions that the default `anon` key does not have for DDL operations like `CREATE EXTENSION`.
    *   If the automatic table/extension creation fails, you might need to:
        1.  Manually run the `CREATE EXTENSION IF NOT EXISTS vector;` command in the SupaBase SQL Editor.
        2.  Consider creating a Supabase database function (e.g., named `exec` that takes a SQL string argument and executes it with definer rights) that the Python client can call with elevated privileges. The `supabase_client.py` file contains comments on this.
        3.  Alternatively, manually create the `crawled_pages` and `page_embeddings` tables using the SQL schema provided in the comments within `utils/supabase_client.py` via the SupaBase SQL Editor.

### Step 4.5: API Key Configuration (Local)
When you first run the Streamlit application, you will be prompted to enter the following in the UI:
*   **SupaBase URL**
*   **SupaBase Anon Key**
*   **Google Gemini API Key**

These keys are stored in Streamlit's session state for your current session and are essential for the application's core functionalities. They are not stored persistently by the application itself beyond the session.

## 5. Running the Application Locally

1.  Ensure your virtual environment is activated.
2.  Navigate to the `seo_app/` directory in your terminal (this directory contains `app.py`).
3.  Run the following command:
    ```bash
    streamlit run app.py
    ```
4.  The application should automatically open in your default web browser. If not, your terminal will display a local URL (usually `http://localhost:8501`) that you can open.

## 6. Using the Application

The application is divided into several sections, accessible via the sidebar navigation.

*   **Home:** Provides a welcome message and overview.

*   **SupaBase & API Key Configuration:**
    *   **SupaBase:** Typically configured in the "Embeddings & SupaBase" section. Enter your SupaBase Project URL and Anon Key. Click "Connect to SupaBase & Ensure Tables" to initialize the connection and attempt to set up necessary database tables.
    *   **Google Gemini API Key:** Input fields for the Gemini API key appear in sections that require it (e.g., "Embedding Generation", "Content Clustering" if LLM naming is enabled). This key is stored in the session state.

*   **Web Crawling:**
    *   **URL to crawl:** The starting URL for the crawl.
    *   **Allowed URL Substring:** (Optional) Restricts crawling to URLs containing this substring.
    *   **Maximum Crawl Depth:** How many links deep to follow from the start URL.
    *   **Maximum Pages to Crawl:** A limit on the total number of pages.
    *   **Max Concurrent Threads:** Number of parallel requests.
    *   **Memory Threshold (%):** Stops crawling if system memory usage exceeds this.
    *   **Processing Mode:** "Stream" (processes URLs as they are found) or "Batch".
    *   **Batch Size:** Number of URLs to process in a batch if "Batch" mode is selected.
    *   **Content Verbose:** More detailed logging from the crawler.
    *   **BM25 Threshold, Min Word Threshold, Require H1:** Parameters for `crawl4ai`'s content filtering strategy to fetch only relevant content.
    *   **Crawler Mode:** "HTTP" (faster, no JS rendering) or "Playwright" (slower, full browser rendering for JS-heavy sites).
    *   **Disable JavaScript (Playwright only):** Can speed up Playwright if JS isn't critical for content.
    *   **Fast Timeouts:** Uses more aggressive network timeouts.
    *   Click "Start Crawling". Progress will be displayed. Results (a table of crawled URLs, titles, content snippets, etc.) can be downloaded as a CSV.

*   **Embeddings & SupaBase:**
    *   **Connect to SupaBase:** First, ensure your SupaBase URL and Key are entered and you've clicked "Connect...".
    *   **Data Source:** Choose to use data from the "Web Crawling" section (if a crawl was just run) or upload a CSV file (must contain at least a 'url' column, and preferably content/metadata columns like 'title', 'main_content_raw', 'meta_description').
    *   **Embedding Configuration:**
        *   Enter your Google Gemini API Key.
        *   Select the Google Embedding Model (e.g., `models/embedding-001`).
        *   Choose to generate "Metadata Embeddings" and/or "Content Embeddings".
        *   For each, select the appropriate "Task Type" (e.g., `RETRIEVAL_DOCUMENT` is common).
        *   Specify maximum tokens for metadata/content input (truncation occurs if exceeded).
        *   For content embeddings, select the DataFrame column containing the main text.
        *   Set "Max Requests Per Minute" to respect API rate limits.
        *   "Force Re-generation": If checked, embeddings will be re-generated even if they already exist in SupaBase for a given page and embedding type.
    *   Click "Start Embedding Generation & Storage". Crawled page data (from the input DataFrame) will be saved to the `crawled_pages` table in SupaBase, and generated embeddings will be stored in the `page_embeddings` table.

*   **Similarity Analysis:**
    *   Ensure SupaBase is connected.
    *   **Select Embedding Type:** Choose the specific embedding type you want to analyze (e.g., `content_yourcolumn_models/embedding-001`). These are dynamically fetched from your `page_embeddings` table.
    *   **Domain Filter (Optional):** Restrict analysis to a single domain.
    *   **Top K Similar Pages:** How many similar pages to list for each URL.
    *   **Minimum Similarity Threshold:** The minimum cosine similarity score to consider a page similar.
    *   Click "Run Similarity Analysis". Results show each source URL and its most similar counterparts with their similarity scores. Downloadable as CSV.

*   **Content Issue Detection:**
    *   This analysis specifically uses **metadata embeddings** to find potential duplicates or cannibalization.
    *   Select the appropriate **Metadata Embedding Type**.
    *   Optionally filter by domain.
    *   **Thresholds:**
        *   `Duplicate Threshold`: (e.g., 0.999) Similarity at or above this indicates likely duplicates.
        *   `Cannibalization Min/Max Threshold`: (e.g., 0.90-0.998) Pages in this similarity range might be competing for the same keywords.
    *   Click "Detect Content Issues". The output lists pairs of URLs, their titles, similarity score, and the identified issue type. An interpretation guide is provided.

*   **Content Clustering:**
    *   Select an **Embedding Type** (content or metadata) for clustering.
    *   Optionally filter by domain.
    *   Choose **Clustering Mode**:
        *   `Flat`: Uses KMeans. Can auto-determine optimal cluster count (k) using Silhouette Score within a specified range (Min/Max Clusters) or use a manually set number of clusters.
        *   `Hierarchical`: Performs two-level clustering (KMeans for super-clusters, then KMeans+Silhouette for sub-clusters within each large enough super-cluster). Configure number of super-clusters and sub-cluster parameters.
    *   **LLM Cluster Naming (Optional):**
        *   If enabled, uses Google Gemini to generate a descriptive name for each cluster based on its content (e.g., from the 'title' column you specify). Requires API Key and model selection.
    *   Click "Run Clustering Analysis". Results include original data plus `cluster_id` and (if enabled) `cluster_name`.
    *   **Cluster Exploration:** View sample URLs/titles within each cluster.
    *   **Visualizations:** Generate interactive PCA and t-SNE scatter plots (using Plotly) to visualize clusters. (Note: t-SNE may be slow for very large datasets).

*   **Topical Authority (Site Focus & Radius):**
    *   This analysis typically uses **content embeddings**. Select the appropriate embedding type.
    *   Optionally filter by domain.
    *   Click "Calculate Topical Authority Metrics".
    *   **Results:**
        *   **Domain-Level:** Shows `AvgSiteRadius` (average distance of pages from their domain's centroid), `PageCount`, and `DomainFocus` ( `1 / (1 + AvgSiteRadius)` - higher is more focused).
        *   **Page-Level (Optional):** Shows `SiteRadius` for each individual page.
    *   **Visualizations:**
        *   Scatter plot: `AvgSiteRadius` vs. `DomainFocus`, sized by `PageCount`.
        *   Histograms: Distribution of `SiteRadius` for the top N domains.

## 7. Output Files

Most analysis sections provide a button to download the results as a CSV file. These files are typically named with the analysis type, embedding type used, and a timestamp, for example:
*   `crawled_data_<timestamp>.csv`
*   `similarity_analysis_<embedding_type>_<timestamp>.csv`
*   `content_issues_<embedding_type>_<timestamp>.csv`
*   `clustering_results_<embedding_type>_<timestamp>.csv`
*   `domain_topical_authority_<embedding_type>_<timestamp>.csv`
*   `page_level_siteradius_<embedding_type>_<timestamp>.csv`

## 8. Deployment

For deploying this application, refer to the `DEPLOYMENT_NETLIFY.md` file in this directory. The recommended approach involves hosting the Streamlit application on a platform suited for Python web apps (like Streamlit Cloud, Heroku, or a container service) and optionally using Netlify for a custom domain or static landing page.

## 9. Troubleshooting/Notes

*   **Processing Times:** Crawling, embedding generation, and some analyses (especially clustering on large datasets) can be time-consuming. Run these on smaller datasets first to estimate processing time.
*   **API Rate Limits:** Be mindful of API rate limits for Google Gemini, especially during bulk embedding generation. The application includes a "requests per minute" setting for this.
*   **SupaBase Performance:** For very large datasets (many tens of thousands of pages/embeddings), query performance from SupaBase might become a consideration. Ensure appropriate indexing on your SupaBase tables (indexes on `embedding_type`, `domain`, `url`, and vector indexes like HNSW or IVFFlat on `embedding_vector` are crucial).
*   **Playwright Browsers:** If using `crawl4ai` in Playwright mode locally and you encounter issues, ensure browser binaries are installed. Running `npx playwright install --with-deps chromium` (or the equivalent for other browsers) in your environment often resolves this. Hosted platforms might handle this automatically.
*   **Resource Limits:** For local execution, ensure your machine has sufficient RAM, especially for Playwright crawling and in-memory Pandas operations. For deployed versions, choose an appropriate plan on your hosting platform.
```
