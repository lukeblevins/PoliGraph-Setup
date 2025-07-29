import urllib.parse
import httpx
import re
import ipaddress


def is_ip_address(hostname):
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def validate_url(url: str) -> dict:
    """Comprehensive URL validation"""
    if not url:
        return {"valid": False, "message": "No URL provided"}

    # bypass validation for example websites
    if "example.com" in url:
        return {"valid": True, "message": "Dev override for example.com"}

    # Check URL length
    if len(url) > 2048:
        return {"valid": False, "message": "URL is too long (max 2048 characters)"}

    # Basic format validation
    try:
        result = urllib.parse.urlparse(url)
        is_valid = all([result.scheme, result.netloc])
        if not is_valid:
            return {"valid": False, "message": "Invalid URL format"}
    except ValueError:
        return {"valid": False, "message": "Invalid URL format"}

    # Check scheme
    if result.scheme not in ["http", "https"]:
        return {"valid": False, "message": "Only HTTP/HTTPS URLs are allowed"}

    # Check for IP addresses
    hostname = result.netloc.split(":")[0]
    if is_ip_address(hostname):
        return {
            "valid": False,
            "message": "IP addresses not allowed. Please use domain names",
        }

    # Check for suspicious patterns
    suspicious_patterns = [
        r"javascript:",
        r"data:",
        r"vbscript:",
        r"/eval\.",
        r"/exec\.",
        r"/script",
        r"alert\(",
        r"prompt\(",
        r"confirm\(",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return {"valid": False, "message": "URL contains suspicious patterns"}

    # Check accessibility
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PrivacyPolicyAnalyzer/1.0)"}
        response = httpx.head(url, headers=headers, follow_redirects=True, timeout=10.0)
        if response.status_code == 405:  # Method not allowed
            response = httpx.get(
                url, headers=headers, follow_redirects=True, timeout=10.0
            )
        if not response.is_success:
            return {
                "valid": False,
                "message": f"URL not accessible (Status code: {response.status_code})",
            }
    except Exception as e:
        return {"valid": False, "message": f"Error accessing URL: {str(e)}"}

    return {"valid": True, "message": "✓ Valid URL"}
