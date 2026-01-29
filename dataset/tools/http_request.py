import requests
import time

def safe_get(url, params=None, headers=None, retries=64, interval=1.0):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            print(f"failed to GET {url} (attempt {i+1}/{retries}): {e}")
            time.sleep(interval)
    return None