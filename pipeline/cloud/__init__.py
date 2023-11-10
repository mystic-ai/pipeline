import urllib.parse

import httpx


def authenticate(token: str, url: str) -> bool:
    auth_url = urllib.parse.urljoin(url, "/v4/tokens/validate")
    response = httpx.get(
        auth_url,
        headers={
            "Authorization": f"Bearer {token}",
        },
    )
    response.raise_for_status()
