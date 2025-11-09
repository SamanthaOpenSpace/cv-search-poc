Here is the updated `README.md` file, edited to include clear, professional instructions for setting up `rclone` locally.

-----

# cv-search

A RAG-based system for parsing, indexing, and searching candidate CVs.

## 🚀 Local Setup

1.  **Create Virtual Environment**

    ```sh
    # Using python 3.11+
    py -3.11 -m venv .venv

    # Activate (Windows PowerShell)
    .\.venv\Scripts\Activate.ps1

    # Activate (macOS/Linux)
    # source .venv/bin/activate
    ```

2.  **Install Dependencies**

    ```sh
    pip install -e .
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory and add your OpenAI API key. If using Azure, add your Azure-specific keys (see `src/cvsearch/settings.py` for all options).

    ```ini
    # For standard OpenAI
    OPENAI_API_KEY=sk-proj-*****************

    # Or for Azure
    # USE_AZURE_OPENAI=True
    # OPENAI_API_KEY=your-azure-api-key
    # AZURE_ENDPOINT=https://your-resource.openai.azure.com/
    # AZURE_API_VERSION=2024-02-01
    # OPENAI_MODEL=your-chat-deployment-name
    # OPENAI_EMBED_MODEL=your-embedding-deployment-name
    ```

4.  **Initialize Database**
    This creates the `cvsearch.db` SQLite file and runs the schema.

    ```sh
    python .\main.py init-db
    ```

5.  **Ingest Mock Data**
    This command clears the database, loads all CVs from `data/mock_cvs.json`, and builds the local FAISS vector index (`data/cv_search.faiss`).

    ```sh
    python .\main.py ingest-mock
    ```

-----

## ☁️ Optional: Setting Up Google Drive Sync

This project uses `rclone` to sync new CVs (as `.pptx` files) from a Google Drive folder. The `ingest-gdrive` command will then parse these files, convert them to JSON, and ingest them into the database.

### 1\. Install `rclone`

`rclone` is a command-line tool for managing files on cloud storage.

* **Official Docs:** Download the binary from the [rclone downloads page](https://rclone.org/downloads/).
* **macOS (Homebrew):** `brew install rclone`
* **Windows (Scoop/Chocolatey):** `scoop install rclone` or `choco install rclone`

After installation, ensure the `rclone` executable is available in your system's `PATH`. You can verify this by running:

```sh
rclone --version
```

### 2\. Configure `rclone` for Google Drive

You need to create a "remote" configuration for Google Drive.

1.  Run the interactive configuration tool:
    ```sh
    rclone config
    ```
2.  Follow the prompts:
    * `n` (New remote)
    * **`name>`**: Enter a name for your remote. We recommend `gdrive`. This name **must** match the `GDRIVE_REMOTE_NAME` in your `.env` file.
    * **`Storage>`**: Find "Google Drive" in the list and enter its corresponding number (e.g., `18`).
    * **`client_id>`**: Press Enter to leave blank.
    * **`client_secret>`**: Press Enter to leave blank.
    * **`scope>`**: Press Enter or type `1` for full access.
    * **`root_folder_id>`**: Press Enter to leave blank (searches all of "My Drive").
    * **`service_account_file>`**: Press Enter to leave blank.
    * **`Edit advanced config?`**: `n` (No)
    * **`Use auto config?`**: `y` (Yes). This will open a browser window.
    * In the browser, log in to the Google account that has access to the CV folder and grant `rclone` permissions.
    * **`Configure this as a team drive?`**: `n` (No), unless your CV folder is in a Shared Drive.
    * **`y/e/d>`**: `y` (Yes, this is OK) to save the configuration.
    * **`q`** (Quit config) to exit.

### 3\. Update `.env` File

Add the following variables to your `.env` file:

```ini
# The 'name' you gave your remote in `rclone config`
GDRIVE_REMOTE_NAME=gdrive

# The folder path *on your Google Drive* to sync from
GDRIVE_SOURCE_DIR=CV_Inbox

# The local folder where CVs will be downloaded
# Default is "data/gdrive_inbox"
GDRIVE_LOCAL_DEST_DIR=data/gdrive_inbox

# (Optional) Path to your rclone.conf file.
# Leave this commented out or blank if rclone is using its default location.
# GDRIVE_RCLONE_CONFIG_PATH=
```

### 4\. Run the Sync and Ingest Commands

Now you can run the CLI commands.

1.  **Sync Files:** Pulls `.pptx` files from Google Drive to your local `GDRIVE_LOCAL_DEST_DIR`.

    ```sh
    python .\main.py sync-gdrive
    ```

2.  **Ingest Files:** Parses the downloaded `.pptx` files, converts them to structured JSON (saved in `data/ingested_cvs_json/`), and upserts them into the database and FAISS index.

    ```sh
    python .\main.py ingest-gdrive
    ```

-----

## 🛠️ Usage

### Run Streamlit UI

The easiest way to use the app is via the Streamlit web interface.

```sh
streamlit run app.py
```

### Run CLI Commands

You can also test individual components from the command line.

**Parse a free-text request into structured JSON:**

```sh
python .\main.py parse-request
```

**Run a single-seat search (using `criteria.json`):**

```sh
python .\main.py search-seat --criteria ./criteria.json --topk 2
```

**Plan a presale team from a brief:**

```sh
python main.py presale-plan --text "Mobile + web app with Flutter/React; AI chatbot for goal setting; partner marks failures; donation on failure."
```

**Run a full project search (deriving seats from text):**

```sh
python main.py project-search --text "Mobile+web app in Flutter/React; AI chatbot; partner marks failures; donation on failure." --topk 3
```

**Run a full project search (using explicit seats from JSON):**

```sh
python main.py project-search --criteria ./criteria.json --topk 3
```