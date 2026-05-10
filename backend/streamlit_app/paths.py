from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
SRC_DIR = BACKEND_DIR / "src"
IMAGES_DIR = SRC_DIR / "images"

CSV_PATH = BACKEND_DIR / "final_data.csv"
EMBEDDINGS_PATH = BACKEND_DIR / "embeddings"
AUTH_DB_PATH = APP_DIR / "auth.db"
HISTORY_DB_PATH = APP_DIR / "search_history.db"
LOGO_PATH = IMAGES_DIR / "logo.png"