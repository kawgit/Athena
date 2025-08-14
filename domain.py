import requests
from concurrent.futures import ThreadPoolExecutor
import time
import string

API_KEY = "3mM44Ywf7afdkg_9zjhdxRLktbVfdwdgkjbk3"
API_SECRET = "revoked"
TLDs = ['ai']  # You can extend this list
MAX_RETRIES = 10
BASE_DELAY = 1  # seconds

def load_domains_from_file(filepath):
    with open(filepath, "r") as f:
        words = [line.strip() for line in f]
        print(f"Loaded {len(words)} words from {filepath}")
        return words

def check_domain_availability(domain):
    time.sleep(3)

    url = f"https://api.ote-godaddy.com/v1/domains/available?domain={domain}"
    headers = {
        "Authorization": f"sso-key {API_KEY}:{API_SECRET}",
        "Accept": "application/json"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return domain, data.get("available", False)
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY * attempt
                print(f"Error checking {domain} (attempt {attempt}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Error checking {domain} (attempt {attempt}): {e}. Giving up.")
                return domain, False

def main():
    base_names = load_domains_from_file("domains.txt")
    domain_combinations = [f"{name}.{tld}" for name in base_names for tld in TLDs]

    available = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_domain_availability, domain) for domain in domain_combinations]

        for future in futures:
            domain, is_available = future.result()
            print(f"{domain}: {'Available' if is_available else 'Taken'}")
            if is_available:
                available.append(domain)

    print("\nAvailable domains:")
    for domain in available:
        print(domain)

if __name__ == "__main__":
    main()
