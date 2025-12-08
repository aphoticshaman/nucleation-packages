"""
Test script for Anthropic, XAI, and OpenAI API keys.

This script tests each API key by making a simple request to verify:
1. The key is valid
2. The API is accessible
3. Basic functionality works

Usage:
    Option 1: Set environment variables:
    - ANTHROPIC_API_KEY
    - XAI_API_KEY (or xai-hydra)
    - OPENAI_API_KEY (or openai-hydra)
    
    Option 2: Pass keys as command-line arguments:
    python test_api_keys.py --anthropic-key YOUR_KEY --xai-key YOUR_KEY --openai-key YOUR_KEY
    
    Option 3: Create a .env file with:
    ANTHROPIC_API_KEY=your_key
    XAI_API_KEY=your_key
    OPENAI_API_KEY=your_key
    
    Then run: python test_api_keys.py
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, that's okay

# Try importing required packages
try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Install with: pip install anthropic")
    anthropic = None

try:
    import openai
except ImportError:
    print("ERROR: openai package not installed. Install with: pip install openai")
    openai = None


def test_anthropic_api(api_key: Optional[str]) -> Dict[str, Any]:
    """Test Anthropic API key."""
    result = {
        "provider": "Anthropic",
        "status": "unknown",
        "error": None,
        "details": {}
    }
    
    if not api_key:
        result["status"] = "missing"
        result["error"] = "ANTHROPIC_API_KEY environment variable not set"
        return result
    
    if anthropic is None:
        result["status"] = "error"
        result["error"] = "anthropic package not installed"
        return result
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Make a simple test request
        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Using a cheaper/faster model for testing
            max_tokens=10,
            messages=[
                {"role": "user", "content": "Say 'test successful' if you can read this."}
            ]
        )
        
        result["status"] = "success"
        result["details"] = {
            "model": "claude-3-haiku-20240307",
            "response_preview": message.content[0].text[:50] if message.content else "No content",
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens
            }
        }
        
    except anthropic.APIError as e:
        result["status"] = "error"
        result["error"] = f"API Error: {str(e)}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


def test_openai_api(api_key: Optional[str]) -> Dict[str, Any]:
    """Test OpenAI API key."""
    result = {
        "provider": "OpenAI",
        "status": "unknown",
        "error": None,
        "details": {}
    }
    
    if not api_key:
        result["status"] = "missing"
        result["error"] = "OPENAI_API_KEY environment variable not set"
        return result
    
    if openai is None:
        result["status"] = "error"
        result["error"] = "openai package not installed"
        return result
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Make a simple test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'test successful' if you can read this."}
            ],
            max_tokens=10
        )
        
        result["status"] = "success"
        result["details"] = {
            "model": "gpt-3.5-turbo",
            "response_preview": response.choices[0].message.content[:50] if response.choices else "No content",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except openai.AuthenticationError as e:
        result["status"] = "error"
        result["error"] = f"Authentication Error: Invalid API key - {str(e)}"
    except openai.APIError as e:
        result["status"] = "error"
        result["error"] = f"API Error: {str(e)}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


def test_xai_api(api_key: Optional[str]) -> Dict[str, Any]:
    """Test XAI (Grok) API key."""
    result = {
        "provider": "XAI",
        "status": "unknown",
        "error": None,
        "details": {}
    }
    
    if not api_key:
        result["status"] = "missing"
        result["error"] = "XAI_API_KEY environment variable not set"
        return result
    
    if openai is None:
        result["status"] = "error"
        result["error"] = "openai package not installed (required for XAI)"
        return result
    
    try:
        # XAI uses OpenAI-compatible API with a different base URL
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        
        # Make a simple test request
        response = client.chat.completions.create(
            model="grok-beta",  # XAI's model name
            messages=[
                {"role": "user", "content": "Say 'test successful' if you can read this."}
            ],
            max_tokens=10
        )
        
        result["status"] = "success"
        result["details"] = {
            "model": "grok-beta",
            "response_preview": response.choices[0].message.content[:50] if response.choices else "No content",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except openai.AuthenticationError as e:
        result["status"] = "error"
        result["error"] = f"Authentication Error: Invalid API key - {str(e)}"
    except openai.APIError as e:
        result["status"] = "error"
        result["error"] = f"API Error: {str(e)}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


def print_result(result: Dict[str, Any]):
    """Print formatted test result."""
    provider = result["provider"]
    status = result["status"]
    
    # Color codes for terminal (works on most terminals)
    colors = {
        "success": "\033[92m",  # Green
        "error": "\033[91m",    # Red
        "missing": "\033[93m",  # Yellow
        "reset": "\033[0m"
    }
    
    color = colors.get(status, colors["reset"])
    reset = colors["reset"]
    
    print(f"\n{color}{'='*60}{reset}")
    print(f"{color}Provider: {provider}{reset}")
    print(f"{color}Status: {status.upper()}{reset}")
    
    if result["error"]:
        print(f"{color}Error: {result['error']}{reset}")
    
    if result["details"]:
        print(f"{color}Details:{reset}")
        for key, value in result["details"].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    print(f"{color}{'='*60}{reset}")


def main():
    """Main test function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Anthropic, XAI, and OpenAI API keys")
    parser.add_argument("--anthropic-key", help="Anthropic API key")
    parser.add_argument("--xai-key", help="XAI API key")
    parser.add_argument("--openai-key", help="OpenAI API key")
    args = parser.parse_args()
    
    print("Testing API Keys...")
    print("=" * 60)
    
    # Get API keys from command-line args first, then environment variables
    # Check for standard names and Cursor-specific names
    anthropic_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    xai_key = args.xai_key or os.getenv("XAI_API_KEY") or os.getenv("xai-hydra") or os.getenv("XAI_HYDRA")
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY") or os.getenv("openai-hydra") or os.getenv("OPENAI_HYDRA")
    
    # Check if any keys are set
    if not any([anthropic_key, xai_key, openai_key]):
        print("\nWARNING: No API keys found in environment variables!")
        print("\nPlease set the following environment variables:")
        print("  - ANTHROPIC_API_KEY")
        print("  - XAI_API_KEY")
        print("  - OPENAI_API_KEY")
        print("\nOn Windows PowerShell:")
        print("  $env:ANTHROPIC_API_KEY='your-key-here'")
        print("  $env:XAI_API_KEY='your-key-here'")
        print("  $env:OPENAI_API_KEY='your-key-here'")
        print("\nOn Windows CMD:")
        print("  set ANTHROPIC_API_KEY=your-key-here")
        print("  set XAI_API_KEY=your-key-here")
        print("  set OPENAI_API_KEY=your-key-here")
        print("\nOn Linux/Mac:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("  export XAI_API_KEY='your-key-here'")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    results = []
    
    # Test each API
    if anthropic_key:
        print("\nTesting Anthropic API...")
        results.append(test_anthropic_api(anthropic_key))
    else:
        print("\nSkipping Anthropic API (key not set)")
        results.append({
            "provider": "Anthropic",
            "status": "missing",
            "error": "ANTHROPIC_API_KEY not set",
            "details": {}
        })
    
    if xai_key:
        print("\nTesting XAI API...")
        results.append(test_xai_api(xai_key))
    else:
        print("\nSkipping XAI API (key not set)")
        results.append({
            "provider": "XAI",
            "status": "missing",
            "error": "XAI_API_KEY not set",
            "details": {}
        })
    
    if openai_key:
        print("\nTesting OpenAI API...")
        results.append(test_openai_api(openai_key))
    else:
        print("\nSkipping OpenAI API (key not set)")
        results.append({
            "provider": "OpenAI",
            "status": "missing",
            "error": "OPENAI_API_KEY not set",
            "details": {}
        })
    
    # Print summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for result in results:
        print_result(result)
    
    # Final summary
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    missing_count = sum(1 for r in results if r["status"] == "missing")
    
    print(f"\n\nFinal Summary:")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"  ⊘ Missing: {missing_count}")
    print(f"  Total: {len(results)}")
    
    if success_count == len(results):
        print("\n🎉 All API keys are working correctly!")
        sys.exit(0)
    elif success_count > 0:
        print("\n⚠️  Some API keys are working, but some have issues.")
        sys.exit(1)
    else:
        print("\n❌ No API keys are working. Please check your keys and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

