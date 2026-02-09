import os

OVERPASS_URL: str = os.getenv(
    "OVERPASS_URL",
    "https://overpass.kumi.systems/api/interpreter"
)

REQUEST_TIMEOUT: float = float(os.getenv("OVERPASS_TIMEOUT", "20"))
REQUEST_RETRIES: int = int(os.getenv("OVERPASS_RETRIES", "3"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
GIGACHAT_API_TOKEN = os.getenv("GIGACHAT_API_TOKEN", "MDE5YmU1NmItYzVhZi03YjUzLTlmNDktNmQ2YzYzMTI4ZjM3OmMzMzRiNGIxLTJjNjYtNGQ3Yy1hMDQyLTQ0ZmZhNzJkOTdhOQ==")