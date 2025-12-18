import requests
import logging

logger = logging.getLogger('Link validation helper')

# Check if the status code is in the 200-399 range (successful or redirected)
def check_link_validity(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code >= 200 and response.status_code < 400:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        logger.info(f"Link validation check failed: {url}")
        return False

# Example usage
# url = "https://www.example.com"
# if check_link_validity(url):
#     print(f"The link {url} is valid.")
# else:
#     print(f"The link {url} is not valid.")
