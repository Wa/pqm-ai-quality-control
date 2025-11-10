"""Test script to verify DuckDuckGo search engine connection for AI Agent tab."""
import requests
import json
from typing import Dict, List, Any


def test_duckduckgo_connection(query: str = "Python programming", timeout: float = 15.0) -> Dict[str, Any]:
    """Test DuckDuckGo search connection using the same method as tool_web_search."""
    
    print(f"Testing DuckDuckGo connection...")
    print(f"Query: {query}")
    print(f"Timeout: {timeout} seconds")
    print("-" * 60)
    
    try:
        # Same parameters as tool_web_search
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "t": "pqm-ai-agent",
        }
        url = "https://duckduckgo.com/"
        
        print(f"Making GET request to: {url}")
        print(f"Parameters: {params}")
        
        resp = requests.get(url, params=params, timeout=timeout)
        
        print(f"\n✓ HTTP Request successful")
        print(f"  Status Code: {resp.status_code}")
        print(f"  Response Headers: {dict(resp.headers)}")
        
        resp.raise_for_status()
        
        print(f"\n✓ HTTP Status OK")
        print(f"  Attempting to parse JSON response...")
        
        data = resp.json()
        
        print(f"✓ JSON parsing successful")
        print(f"  Response keys: {list(data.keys())}")
        
        # Extract results similar to tool_web_search
        results: List[Dict[str, str]] = []
        
        def _extract(items: List[Dict[str, object]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if "FirstURL" in item and "Text" in item:
                    results.append({
                        "title": str(item.get("Text", "")),
                        "url": str(item.get("FirstURL", "")),
                    })
                if "Topics" in item and isinstance(item.get("Topics"), list):
                    _extract(item["Topics"])
        
        related = data.get("RelatedTopics") or []
        if isinstance(related, list):
            _extract(related)
        
        abstract_text = data.get("AbstractText")
        abstract_url = data.get("AbstractURL")
        if abstract_text and abstract_url:
            results.insert(0, {"title": str(abstract_text), "url": str(abstract_url)})
        
        print(f"\n✓ Search results extracted")
        print(f"  Total results found: {len(results)}")
        
        if results:
            print(f"\n  Sample results:")
            for i, result in enumerate(results[:3], 1):
                print(f"    {i}. {result.get('title', 'N/A')[:80]}...")
                print(f"       URL: {result.get('url', 'N/A')}")
        
        return {
            "success": True,
            "query": query,
            "status_code": resp.status_code,
            "results_count": len(results),
            "results": results[:5],
            "raw_data_keys": list(data.keys()),
            "message": "Connection test successful!"
        }
        
    except requests.exceptions.Timeout as e:
        error_msg = f"Connection timeout after {timeout} seconds"
        print(f"\n✗ {error_msg}")
        print(f"  Error: {str(e)}")
        return {
            "success": False,
            "error_type": "Timeout",
            "error": error_msg,
            "details": str(e)
        }
        
    except requests.exceptions.ConnectionError as e:
        error_msg = "Connection error - unable to reach DuckDuckGo"
        print(f"\n✗ {error_msg}")
        print(f"  Error: {str(e)}")
        print(f"\n  Possible causes:")
        print(f"    - No internet connection")
        print(f"    - Firewall blocking the connection")
        print(f"    - Proxy settings required")
        print(f"    - DNS resolution failure")
        return {
            "success": False,
            "error_type": "ConnectionError",
            "error": error_msg,
            "details": str(e)
        }
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error: {resp.status_code if 'resp' in locals() else 'Unknown'}"
        print(f"\n✗ {error_msg}")
        print(f"  Error: {str(e)}")
        return {
            "success": False,
            "error_type": "HTTPError",
            "error": error_msg,
            "details": str(e),
            "status_code": resp.status_code if 'resp' in locals() else None
        }
        
    except json.JSONDecodeError as e:
        error_msg = "Failed to parse JSON response"
        print(f"\n✗ {error_msg}")
        print(f"  Error: {str(e)}")
        if 'resp' in locals():
            print(f"  Response text (first 500 chars): {resp.text[:500]}")
        return {
            "success": False,
            "error_type": "JSONDecodeError",
            "error": error_msg,
            "details": str(e)
        }
        
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}"
        print(f"\n✗ {error_msg}")
        print(f"  Error: {str(e)}")
        import traceback
        print(f"\n  Traceback:")
        print(traceback.format_exc())
        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": error_msg,
            "details": str(e)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("DuckDuckGo Search Engine Connection Test")
    print("=" * 60)
    print()
    
    # Test with a simple query
    result = test_duckduckgo_connection("Python programming")
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("success"):
        print("\n✓ Connection test PASSED - DuckDuckGo is accessible")
        exit(0)
    else:
        print(f"\n✗ Connection test FAILED - {result.get('error', 'Unknown error')}")
        exit(1)