import re
from typing import List

import requests


class URLWrapper:

    @classmethod
    def embed_icon_for_url_markdown(cls, url: str, tooltip: str = None):
        image = None
        if "sharepoint.com" in url:
            image = "/linkicon/sharepoint.svg"
        elif "service-now.com" in url:
            image = "/linkicon/servicenow.svg"
        elif "nvidia.com/phonedirectory" in url:
            image = "/linkicon/phonebook.png"
        elif "confluence.nvidia.com" in url:
            image = "/linkicon/confluence.svg"
        elif "nvidia.com" in url:
            image = "/linkicon/nvidialogo.png"
        elif "sec.gov" in url:
            image = "/linkicon/domain.png"
        elif "google.com" in url:
            image = "/linkicon/gdrive.svg"
        elif "nvcalendar" in url:
            image = "/linkicon/calendar.png"

        if not image:
            return ""
        if tooltip:
            return f'[embedInlineImage]({image} # tooltip)'

        return f'[embedInlineImage]({image})'

    @classmethod
    def remove_embed_inline_images(cls, text: str) -> str:
        """
        Removes all embedInlineImage patterns from the given text.
        Returns:
            str: The cleaned text with all embedInlineImage patterns removed.
        """
        # Regular expression to match the embedInlineImage pattern
        pattern = r'\[embedInlineImage\]\(.*?(?:# .*?)?\)'

        # Remove all matches of the pattern
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text.strip()

    @classmethod
    def extract_links(cls, text: str) -> List[str]:
        # Regex pattern to find markdown formatted URLs
        url_pattern = r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)'

        # Find all matches of markdown links
        urls = re.findall(url_pattern, text)

        unique_urls = {url for _, url in urls}
        return list(unique_urls)

    @classmethod
    def validate_urls(cls, markdown_text: str):
        unique_urls = URLWrapper.extract_links(markdown_text)
        errors = []

        for url in unique_urls:
            try:
                # Send a GET request to check the URL status code
                response = requests.get(url, timeout=5)
                if "sharepoint.com" not in url and response.status_code != 200:
                    errors.append(
                        f"{url} | Error status code: {response.status_code}, information: {response.text}") if response.text else errors.append(
                        f"{url} | Error status code: {response.status_code}")
            except requests.RequestException as e:
                # Append errors for failed requests (e.g., timeout or connection error)
                errors.append(f"{url} | Error: {str(e)}")

        return errors if errors else None
