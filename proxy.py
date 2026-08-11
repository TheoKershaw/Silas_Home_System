"""
Silas proxy server
-------------------
Fetches a target site server-side, strips the headers that block iframing
(X-Frame-Options, Content-Security-Policy: frame-ancestors), and rewrites
relative asset URLs (href/src/srcset/action, and url(...) in inline CSS)
to absolute URLs so the page still renders correctly inside your iframe.

Run alongside your existing Flask app (or merge these routes into it).

    pip install flask requests beautifulsoup4 --break-system-packages
    python proxy_server.py

Then point your mini-window iframe at:
    /proxy?url=https://www.gbnews.com/

NOTE: this is fine for a personal/local project. It is rehosting someone
else's content through your server, so don't deploy this publicly or use
it commercially without checking that site's terms of service.
"""

from flask import Flask, request, Response
from urllib.parse import urljoin, urlparse
import requests
import re

app = Flask(__name__)

# Only allow proxying these hosts — stops this route being used as an
# open relay for arbitrary URLs.
ALLOWED_HOSTS = {
    "www.gbnews.com",
    "gbnews.com",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# Response headers that block framing / cause mismatches — strip these
# before we send the page back to the browser.
STRIP_HEADERS = {
    "x-frame-options",
    "content-security-policy",
    "content-security-policy-report-only",
    "content-encoding",   # we already decompressed via requests
    "content-length",     # body length changes after we rewrite it
    "transfer-encoding",
    "connection",
}

ATTR_URL_TAGS = [
    (re.compile(r'(<img[^>]+src=["\'])(/[^"\']*)(["\'])', re.I), True),
    (re.compile(r'(<script[^>]+src=["\'])(/[^"\']*)(["\'])', re.I), True),
    (re.compile(r'(<link[^>]+href=["\'])(/[^"\']*)(["\'])', re.I), True),
    (re.compile(r'(<a[^>]+href=["\'])(/[^"\']*)(["\'])', re.I), True),
]


def rewrite_relative_urls(html: str, base_url: str) -> str:
    """Rewrite root-relative URLs (starting with /) to absolute URLs
    pointing at the original site, so CSS/JS/images/links keep working."""

    def repl(match):
        prefix, path, suffix = match.group(1), match.group(2), match.group(3)
        return prefix + urljoin(base_url, path) + suffix

    for pattern, _ in ATTR_URL_TAGS:
        html = pattern.sub(repl, html)

    # inline CSS url(/foo.png)
    html = re.sub(
        r'url\((["\']?)(/[^)"\']*)\1\)',
        lambda m: f'url({m.group(1)}{urljoin(base_url, m.group(2))}{m.group(1)})',
        html,
        flags=re.I,
    )

    return html


@app.route("/proxy")
def proxy():
    target = request.args.get("url", "")
    if not target:
        return Response("Missing ?url=", status=400)

    parsed = urlparse(target)
    if parsed.hostname not in ALLOWED_HOSTS:
        return Response("Host not allowed", status=403)

    try:
        upstream = requests.get(target, headers=HEADERS, timeout=10)
    except requests.RequestException as e:
        return Response(f"Upstream fetch failed: {e}", status=502)

    content_type = upstream.headers.get("Content-Type", "")

    if "text/html" in content_type:
        body = rewrite_relative_urls(upstream.text, target)
    else:
        body = upstream.content

    out_headers = [
        (k, v) for k, v in upstream.headers.items()
        if k.lower() not in STRIP_HEADERS
    ]
    # explicitly allow being framed by your own page
    out_headers.append(("X-Frame-Options", "ALLOWALL"))  # ignored by modern browsers but harmless
    out_headers.append(("Content-Security-Policy", "frame-ancestors *"))

    return Response(body, status=upstream.status_code, headers=out_headers)


if __name__ == "__main__":
    app.run(port=5001, debug=True)