#!/usr/bin/env python3
"""
Alternative accuracy test script for vLLM using direct API calls
Calculates perplexity on wikitext dataset without using lm_eval
"""

import requests
import json
import numpy as np
import sys
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
import math

def download_wikitext():
    """Download wikitext dataset"""
    print("INFO: Downloading wikitext dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join([item['text'] for item in dataset if item['text'].strip()])
        return text
    except Exception as e:
        print(f"ERROR: Failed to download wikitext: {e}")
        print("INFO: Attempting to install datasets library...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join([item['text'] for item in dataset if item['text'].strip()])
        return text

def tokenize_text(text: str, tokenizer) -> List[int]:
    """Tokenize text using the tokenizer"""
    return tokenizer.encode(text)

def get_tokenizer(model_name: str):
    """Get the tokenizer for the model"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        print("INFO: Attempting to install transformers library...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer

def call_vllm_api(base_url: str, prompt: str, max_tokens: int = 1, 
                  logprobs: int = 1, temperature: float = 0.0, 
                  api_endpoint: str = None, model_name: str = None) -> Dict:
    """Call vLLM completion API"""
    # Try different API endpoints if not specified
    endpoints_to_try = []
    if api_endpoint:
        endpoints_to_try = [api_endpoint]
    else:
        endpoints_to_try = [
            "/v1/completions",
        ]
    
    # Build payload - use model name if provided, otherwise use a placeholder
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": logprobs,
        "echo": True,  # Important: return logprobs for prompt tokens
    }
    
    # Add model field if provided
    if model_name:
        payload["model"] = model_name
    
    last_error = None
    for endpoint in endpoints_to_try:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            # Success! Save this endpoint for future calls
            return response.json(), endpoint
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Try next endpoint
                last_error = e
                continue
            else:
                # Other HTTP error, don't retry
                print(f"ERROR: API call failed at {url}: {e}")
                return None, None
        except requests.exceptions.RequestException as e:
            print(f"ERROR: API call failed at {url}: {e}")
            return None, None
    
    # All endpoints failed
    print(f"ERROR: All API endpoints failed. Last error: {last_error}")
    print(f"Tried endpoints: {endpoints_to_try}")
    return None, None

def calculate_perplexity_sliding_window(text: str, tokenizer, base_url: str, 
                                       max_length: int = 2048, stride: int = 512,
                                       api_endpoint: str = None, model_name: str = None) -> Tuple[float, float]:
    """
    Calculate perplexity using sliding window approach
    Returns: (token_perplexity, bits_per_byte)
    """
    print(f"INFO: Tokenizing text...")
    encodings = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"INFO: Total tokens: {len(encodings)}")
    print(f"INFO: Using sliding window with max_length={max_length}, stride={stride}")
    
    seq_len = len(encodings)
    nlls = []  # negative log likelihoods
    prev_end_loc = 0
    
    # Progress bar
    num_windows = (seq_len - max_length) // stride + 1
    pbar = tqdm(total=num_windows, desc="Calculating perplexity")
    
    # Use provided endpoint or detect on first call
    working_endpoint = api_endpoint
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # Target length for calculating NLL
        
        input_ids = encodings[begin_loc:end_loc]
        
        # Decode tokens to text for API call
        input_text = tokenizer.decode(input_ids)
        
        # Call vLLM API to get logprobs
        result, detected_endpoint = call_vllm_api(base_url, input_text, max_tokens=1, 
                                                   logprobs=1, api_endpoint=working_endpoint,
                                                   model_name=model_name)
        
        # Save the working endpoint for subsequent calls
        if detected_endpoint and not working_endpoint:
            working_endpoint = detected_endpoint
            print(f"\nINFO: Using API endpoint: {working_endpoint}")
        
        if result is None:
            print("\nERROR: API call failed, skipping this window")
            continue
        
        # Extract logprobs from response
        try:
            choices = result.get('choices', [])
            if not choices:
                print("\nWARNING: No choices in response, skipping")
                continue
            
            logprobs_data = choices[0].get('logprobs', {})
            token_logprobs = logprobs_data.get('token_logprobs', [])
            
            if not token_logprobs:
                print("\nWARNING: No logprobs in response, skipping")
                continue
            
            # Skip the first token (it has no logprob) and use only the target length
            # We calculate NLL for the new tokens only (not already processed)
            target_logprobs = token_logprobs[max(0, len(token_logprobs) - trg_len):]
            
            # Filter out None values
            target_logprobs = [lp for lp in target_logprobs if lp is not None]
            
            if target_logprobs:
                neg_log_likelihood = -sum(target_logprobs)
                nlls.append(neg_log_likelihood)
        
        except Exception as e:
            print(f"\nERROR: Failed to process response: {e}")
            continue
        
        prev_end_loc = end_loc
        pbar.update(1)
        
        if end_loc == seq_len:
            break
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.01)
    
    pbar.close()
    
    if not nlls:
        print("ERROR: No valid perplexity values calculated")
        return None, None
    
    # Calculate perplexity
    total_nll = sum(nlls)
    ppl = math.exp(total_nll / seq_len)
    
    # Calculate bits per byte
    text_bytes = len(text.encode('utf-8'))
    bits_per_byte = total_nll / (text_bytes * math.log(2))
    
    return ppl, bits_per_byte

def calculate_perplexity_simple(text: str, tokenizer, base_url: str, 
                                chunk_size: int = 512, api_endpoint: str = None,
                                model_name: str = None) -> Tuple[float, float, float]:
    """
    Calculate perplexity using simple chunking
    Returns: (token_perplexity, byte_perplexity, bits_per_byte)
    """
    print(f"INFO: Tokenizing text...")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"INFO: Total tokens: {len(tokens)}")
    print(f"INFO: Chunk size: {chunk_size}")
    
    total_log_likelihood = 0.0
    total_tokens = 0
    
    # Split into chunks
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    print(f"INFO: Processing {num_chunks} chunks...")
    
    # Use provided endpoint or detect on first call
    working_endpoint = api_endpoint
    
    for i in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Decode chunk to text
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Call vLLM API
        result, detected_endpoint = call_vllm_api(base_url, chunk_text, max_tokens=1, 
                                                   logprobs=1, api_endpoint=working_endpoint,
                                                   model_name=model_name)
        
        # Save the working endpoint for subsequent calls
        if detected_endpoint and not working_endpoint:
            working_endpoint = detected_endpoint
            print(f"\nINFO: Using API endpoint: {working_endpoint}")
        
        if result is None:
            print(f"\nWARNING: API call failed for chunk {i}, skipping")
            continue
        
        try:
            choices = result.get('choices', [])
            if not choices:
                continue
            
            logprobs_data = choices[0].get('logprobs', {})
            token_logprobs = logprobs_data.get('token_logprobs', [])
            
            # Filter out None values (first token usually has None)
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            
            if valid_logprobs:
                chunk_log_likelihood = sum(valid_logprobs)
                total_log_likelihood += chunk_log_likelihood
                total_tokens += len(valid_logprobs)
        
        except Exception as e:
            print(f"\nERROR: Failed to process chunk {i}: {e}")
            continue
        
        # Small delay
        time.sleep(0.01)
    
    if total_tokens == 0:
        print("ERROR: No valid tokens processed")
        return None, None, None
    
    # Calculate word perplexity (token-based)
    word_perplexity = math.exp(-total_log_likelihood / total_tokens)
    
    # Calculate byte perplexity
    text_bytes = len(text.encode('utf-8'))
    byte_perplexity = math.exp(-total_log_likelihood / text_bytes)
    
    # Calculate bits per byte
    bits_per_byte = -total_log_likelihood / (text_bytes * math.log(2))
    
    return word_perplexity, byte_perplexity, bits_per_byte

def check_server_health(base_url: str) -> bool:
    """Check if vLLM server is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def explore_server_endpoints(base_url: str, model_name: str = None):
    """Explore and print available server endpoints"""
    print("\n" + "="*60)
    print("Exploring vLLM server endpoints...")
    print("="*60)
    
    # Try common endpoints to see what's available
    test_endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/v1/models", "GET"),
        ("/models", "GET"),
        ("/docs", "GET"),
        ("/openapi.json", "GET"),
        ("/v1/completions", "GET"),
        ("/v1/completions", "POST"),
        ("/v1/chat/completions", "POST"),
    ]
    
    available_endpoints = []
    
    for endpoint, method in test_endpoints:
        url = f"{base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            else:
                # Send proper payload for POST
                test_payload = {
                    "prompt": "test",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 1,
                    "echo": True,
                }
                if model_name:
                    test_payload["model"] = model_name
                response = requests.post(url, json=test_payload, timeout=5)
            
            status = response.status_code
            if status != 404:
                available_endpoints.append((endpoint, method, status))
                print(f"✓ {method:4s} {endpoint:30s} -> {status}")
            else:
                print(f"✗ {method:4s} {endpoint:30s} -> 404 Not Found")
        except Exception as e:
            print(f"✗ {method:4s} {endpoint:30s} -> Error: {str(e)[:40]}")
    
    print("="*60)
    
    if available_endpoints:
        print(f"\nFound {len(available_endpoints)} available endpoint(s)")
        return available_endpoints
    else:
        print("\nWARNING: No available endpoints found!")
        return []

def detect_api_endpoint(base_url: str, model_name: str = None) -> str:
    """Detect which API endpoint is available"""
    endpoints = [
        "/v1/completions",
    ]
    
    test_payload = {
        "prompt": "test",
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 1,
        "echo": True,
    }
    
    # Add model if provided
    if model_name:
        test_payload["model"] = model_name
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            print(f"  Testing {endpoint}...", end=" ")
            response = requests.post(url, json=test_payload, timeout=10)
            print(f"HTTP {response.status_code}")
            if response.status_code != 404:
                if response.status_code == 200 or response.status_code == 422 or response.status_code == 400:
                    print(f"SUCCESS: Detected working API endpoint: {endpoint}")
                    return endpoint
                else:
                    print(f"  Endpoint exists but returned {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("WARNING: Could not detect API endpoint, will try all options")
    return None

def main():
    parser = argparse.ArgumentParser(description="Alternative accuracy test for vLLM")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8888",
                       help="Base URL of vLLM server (default: http://0.0.0.0:8888)")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name for tokenizer")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Chunk size for processing (default: 512)")
    parser.add_argument("--max-chunks", type=int, default=None,
                       help="Maximum number of chunks to process (for testing)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Alternative Accuracy Test for vLLM")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}")
    print("=" * 60)
    print()
    
    # Check server health
    print("INFO: Checking server health...")
    if not check_server_health(args.base_url):
        print("ERROR: vLLM server is not responding")
        print(f"ERROR: Please ensure server is running at {args.base_url}")
        return 1
    print("SUCCESS: Server is healthy")
    print()
    
    # Explore available endpoints
    available = explore_server_endpoints(args.base_url, args.model)
    print()
    
    # Don't fail if exploration didn't work perfectly
    # The important thing is that detect_api_endpoint works
    
    # Detect API endpoint
    print("INFO: Detecting completion API endpoint...")
    api_endpoint = detect_api_endpoint(args.base_url, args.model)
    if api_endpoint:
        print(f"SUCCESS: Will use endpoint: {api_endpoint}")
    else:
        print("ERROR: Could not detect working completion endpoint")
        print("None of the standard completion endpoints are working.")
        return 1
    print()
    
    # Download dataset
    try:
        text = download_wikitext()
        print(f"INFO: Downloaded wikitext dataset ({len(text)} characters)")
        print()
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        return 1
    
    # Get tokenizer
    try:
        print("INFO: Loading tokenizer...")
        tokenizer = get_tokenizer(args.model)
        print(f"SUCCESS: Tokenizer loaded")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        return 1
    
    # Calculate perplexity
    try:
        word_ppl, byte_ppl, bits_per_byte = calculate_perplexity_simple(
            text, tokenizer, args.base_url, args.chunk_size, api_endpoint, args.model
        )
        
        if word_ppl is None:
            print("ERROR: Perplexity calculation failed")
            return 1
        
        print()
        print("=" * 60)
        print("Accuracy Metrics")
        print("=" * 60)
        print(f"bits_per_byte: {bits_per_byte:.4f}")
        print(f"byte_perplexity: {byte_ppl:.4f}")
        print(f"word_perplexity: {word_ppl:.4f}")
        print("=" * 60)
        
        # Also print in the format expected by the benchmark script
        print()
        print("Metrics for parsing:")
        print(f"  'bits_per_byte': {bits_per_byte}")
        print(f"  'byte_perplexity': {byte_ppl}")
        print(f"  'word_perplexity': {word_ppl}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\nINFO: Interrupted by user")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

