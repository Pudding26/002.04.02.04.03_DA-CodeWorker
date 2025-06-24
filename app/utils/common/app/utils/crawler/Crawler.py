import requests
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import time


class Crawler:
    @staticmethod
    def fetch_image_from_url(url: str, retries: int = 5, delay: float = 2.0, timeout: float = 10.0) -> np.ndarray:
        """
        Fetch image from a URL with retries and backoff in case of instability.
        """
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return np.array(Image.open(BytesIO(response.content)))
            
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as e:
                logging.warning(f"⚠️ Attempt {attempt}/{retries} — Failed to fetch image from {url}: {type(e).__name__}: {e}")
                if attempt < retries:
                    time.sleep(delay * attempt)  # Exponential backoff
                else:
                    logging.error(f"🛑 Failed to fetch {url} after {retries} attempts: {e}")
                    raise

            except Exception as e:
                logging.error(f"🛑 Unexpected error fetching {url}: {e}")
                raise