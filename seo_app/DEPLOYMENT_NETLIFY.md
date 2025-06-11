# Deploying the SEO Embeddings Streamlit App (with Netlify Considerations)

## 1. Introduction

Streamlit applications are Python programs that run a web server to provide their interactive UI. This makes them stateful. Netlify, on the other hand, is traditionally optimized for deploying static websites (HTML, CSS, JS) and serverless functions (which are typically stateless and have execution time limits).

Deploying a full, long-running Streamlit application *directly* onto Netlify's standard build/function environment can be complex and may face limitations, especially for:
-   **Long-running processes:** Crawling, embedding generation for many pages, and extensive analysis can exceed serverless function timeouts.
-   **Statefulness:** Streamlit's session state and interactive nature rely on a persistent server process.
-   **Resource Limits:** Serverless functions might have memory/CPU constraints unsuitable for heavy computations.

This guide outlines recommended approaches for deploying this application, considering Netlify's strengths.

## 2. Recommended Approach: External Hosting for Streamlit + Netlify for Frontend/Domain

This is the most robust and straightforward way to deploy this Streamlit application and use Netlify for what it excels at (custom domains, CDN, potentially a static landing page).

### Step 2.1: Deploy the Streamlit App

Choose a platform designed to host Python web applications or containerized services. Here are some popular options:

*   **Streamlit Cloud:** The easiest and most integrated platform for deploying Streamlit apps.
*   **Heroku:** Uses the `Procfile` and `requirements.txt` in your `seo_app/` directory.
*   **Google Cloud Run:** Containerize the Streamlit app (using a `Dockerfile`) and deploy it as a serverless container.
*   **AWS Elastic Beanstalk or ECS/Fargate:** Similar to Cloud Run, involves containerizing the app.
*   **DigitalOcean App Platform:** Offers direct deployment from Git and can run Python apps.

**Generic Deployment Steps for these Platforms:**

1.  **Sign Up:** Create an account on your chosen platform.
2.  **Connect Git Repository:** Link the Git repository containing the `seo_app/` directory to the platform. Ensure the platform is configured to look inside the `seo_app/` subdirectory if your repo has other content at the root.
3.  **Set Environment Variables:** This is crucial. In your chosen platform's settings dashboard for your application, set the following environment variables:
    *   `SUPABASE_URL`: Your SupaBase project URL.
    *   `SUPABASE_KEY`: Your SupaBase Anon Key (public key).
    *   `GEMINI_API_KEY`: Your Google Gemini API Key.
    *   `PYTHONPATH`: If your platform needs to know where to find the `utils` modules relative to `app.py`, you might need to set `PYTHONPATH` to `/app` (or the equivalent root of your `seo_app` directory in the deployed environment). Many platforms handle this automatically if `app.py` is at the root of the deployment.
4.  **Configuration Files:**
    *   Ensure `seo_app/requirements.txt` is complete and lists all dependencies with appropriate versions.
    *   If using Heroku or a similar service, ensure `seo_app/Procfile` (e.g., `web: streamlit run app.py`) is correct.
    *   Ensure `seo_app/runtime.txt` (e.g., `python-3.11.7`) specifies your desired Python version if the platform uses it.
5.  **Build & Deploy:**
    *   Follow the platform's instructions to trigger a build and deploy your application.
    *   The platform will typically install dependencies from `requirements.txt` and then run the command from your `Procfile` (or a similar startup command you configure).
    *   For Streamlit, the command is usually `streamlit run app.py --server.port $PORT --server.headless true`. The `$PORT` environment variable is often provided by the platform.
6.  **Obtain Public URL:** Once deployed, the platform will provide a public URL for your Streamlit application (e.g., `your-app-name.streamlit.app`, `your-app.herokuapp.com`, `your-service.run.app`).

### Step 2.2: Use Netlify for Custom Domain / CDN / Landing Page (Optional)

If you want to use a custom domain, leverage Netlify's CDN, or have a static landing page for your project:

1.  **Create a Static Landing Page (Optional):**
    *   In your repository (or a separate one), create a simple `index.html` and any other static assets (CSS, images) for a landing page. Place these in a directory, e.g., `static_site/`.
2.  **Deploy Static Site to Netlify:**
    *   Connect your Git repository to Netlify.
    *   Configure the build settings:
        *   **Base directory:** (If your static site is in a subdirectory) e.g., `static_site/`
        *   **Build command:** (If you have a build step for your static site, e.g., Jekyll, Hugo, or just need to copy files) e.g., `echo "No build needed"` or your static site generator's build command.
        *   **Publish directory:** The directory containing your `index.html` and other static assets (e.g., `static_site/` or your static site generator's output folder like `_site`, `public`, `dist`).
3.  **Configure Custom Domain:** Set up your custom domain in the Netlify dashboard for this static site.
4.  **Link to or Redirect to Streamlit App:**
    *   **Option A (Simple Link):** Your static landing page can simply have a button or link that directs users to the externally hosted Streamlit app URL obtained in Step 2.1.
    *   **Option B (Netlify Redirect/Proxy):** If you want the Streamlit app to appear as if it's under your custom domain (e.g., `yourdomain.com/app`), you can use Netlify redirects. Add this to your `netlify.toml` (ensure this file is in the base of the repository Netlify is building from, or the "Base directory" if specified):

        ```toml
        # Example netlify.toml for redirecting /app to an external Streamlit app
        [build]
          command = "echo 'Building static landing page...'" # Or your static site build command
          publish = "static_site" # Directory with your index.html

        [[redirects]]
          from = "/app/*"
          to = "https://YOUR_EXTERNAL_STREAMLIT_APP_URL/:splat" # Replace with actual Streamlit app URL
          status = 200 # 200 for proxying (masks URL), 301/302 for permanent/temporary redirect
          force = true # Ensures this rule takes precedence
        ```
        *Note on `status = 200` (proxying): While this makes the Streamlit app appear under your Netlify domain, it can sometimes lead to issues with WebSockets or other stateful interactions if not handled carefully by Netlify's proxy. A simple link or a 301/302 redirect is often more reliable for fully external apps.*

## 3. Alternative: Deploying Python Web Apps on Netlify

Netlify has been evolving its support for backend services. While challenging for a full, stateful Streamlit app, here are some considerations:

### Step 3.1: Using Netlify Serverless Functions for Backend Tasks (Hybrid Model)

This approach involves a significant re-architecture of the current application:
*   **Frontend:** The Streamlit UI would need to be either a static site generator output (if possible to make parts of it static) or a JavaScript framework (React, Vue, etc.).
*   **Backend:** Your Python logic from the `utils/` directory (crawling, embedding generation, analysis functions) would be wrapped in API endpoints using a lightweight framework like Flask or FastAPI. These API endpoints would then be deployed as Netlify Serverless Functions.
*   **Communication:** The frontend would make API calls to these Netlify Functions.
*   **Limitations:**
    *   **Execution Time:** Netlify Functions have execution time limits (e.g., 10 seconds to potentially 26 seconds on paid plans, but can be up to 15 minutes for background functions). Long tasks like crawling or embedding large datasets would need to be broken down, run asynchronously, or use background functions.
    *   **State:** Serverless functions are generally stateless. Streamlit's inherent statefulness would be lost for the direct UI components if not hosted on a proper Streamlit server.
*   The `netlify.toml` in your `seo_app/` directory includes a placeholder `functions = "netlify/functions"` if you were to create your serverless functions in that subfolder.

This is a complex undertaking and changes the nature of the application from an integrated Streamlit app to a JAMstack architecture with a Python serverless backend.

### Step 3.2: Netlify for Python Web Apps (e.g., using Web Standard Adapters)

Netlify is improving its ability to host Python web applications that adhere to WSGI/ASGI standards (like Flask, Django, FastAPI).
*   **Wrapping Streamlit:** It might be theoretically possible to wrap a Streamlit application to make it compatible if a suitable adapter or custom server script can present it as a WSGI/ASGI app that Netlify's Python runtime can serve. This is not a standard Streamlit deployment method and would require significant research and experimentation.
*   **Build Command:** The `netlify.toml` build command `streamlit run app.py` is unlikely to work as a persistent server in Netlify's typical build/runtime environment. Netlify usually expects to be given an application object from a file (e.g., `main:app`).
*   **Important Note:** This route for a full Streamlit app needs careful testing with Netlify's current Python web app support, especially concerning:
    *   **Statefulness:** How Streamlit's session state is handled.
    *   **WebSockets:** Streamlit relies heavily on WebSockets for interactivity.
    *   **Long-running processes:** Background tasks initiated by the app.
    *   **Resource limits.**

    Direct deployment of Streamlit apps is generally better supported by platforms like Streamlit Cloud, Heroku, or container services.

## 4. General Deployment Notes

*   **Environment Variables:** **Crucial!** Always set `SUPABASE_URL`, `SUPABASE_KEY`, and `GEMINI_API_KEY` as environment variables in your chosen deployment platform's settings dashboard. **Do NOT hardcode them in your Python files.**
*   **SupaBase Setup:**
    *   Ensure your SupaBase project is active.
    *   Enable the `vector` extension in your SupaBase database (SQL Editor: `CREATE EXTENSION IF NOT EXISTS vector;`).
    *   The `ensure_tables_exist` function in `utils/supabase_client.py` attempts to create tables. If it fails (especially for enabling extensions due to permissions), you might need to:
        *   Create the `exec` RPC function in SupaBase as described in `supabase_client.py` comments to allow the script to run DDL for table creation.
        *   Or, manually create the `crawled_pages` and `page_embeddings` tables using the SQL schema provided in `supabase_client.py` via the SupaBase SQL Editor.
*   **Python Version:** Ensure your deployment platform uses the Python version specified in `seo_app/runtime.txt` (e.g., `python-3.11.7`) or configure it accordingly in the platform's settings.
*   **Resource Allocation:** For tasks like crawling, embedding generation, and clustering, be mindful of the resource limits (CPU, memory, execution time) of your chosen hosting plan. For serverless functions, long-running tasks are particularly problematic and might require background workers or a different architecture. Streamlit Cloud and container-based solutions often offer more flexible resource allocation.
*   **`requirements.txt`:** Double-check that all necessary packages are included and correctly versioned in `seo_app/requirements.txt`. In a local environment, after installing all dependencies, run `pip freeze > requirements.txt` to get exact versions.
*   **`playwright install`:** Remember that Playwright (a dependency of `crawl4ai`) requires browser binaries to be installed. Some platforms might do this automatically if they detect Playwright in `requirements.txt` (e.g., Streamlit Cloud). For others (like Heroku or custom Docker containers), you might need to add commands to your build process or Dockerfile to run `playwright install` or `playwright install chromium`.
```
