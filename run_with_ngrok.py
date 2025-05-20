import subprocess
import time
import webbrowser
import os
import requests
import json
from app import app

def wait_for_ngrok(max_retries=10, retry_delay=1):
    """Wait for ngrok to be ready"""
    for i in range(max_retries):
        try:
            # Try both old and new API endpoints
            try:
                response = requests.get('http://localhost:4040/api/tunnels')
            except requests.exceptions.ConnectionError:
                response = requests.get('http://127.0.0.1:4040/api/tunnels')
                
            if response.status_code == 200:
                return response.json()
            print(f"Attempt {i+1}: Got status code {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            print(f"Attempt {i+1}: Connection error - {str(e)}")
            if i < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None

def run_ngrok():
    """Run ngrok in a subprocess"""
    print("Starting ngrok...")
    
    # Kill any existing ngrok processes
    os.system('pkill ngrok')
    time.sleep(1)  # Give time for the process to be killed
    
    # Start ngrok with explicit configuration
    try:
        # First, check if ngrok is authenticated
        auth_check = subprocess.run(['ngrok', 'config', 'check'], 
                                  capture_output=True, 
                                  text=True)
        if "not found" in auth_check.stdout or "not found" in auth_check.stderr:
            print("Error: ngrok is not authenticated. Please run 'ngrok authtoken YOUR_TOKEN'")
            print("You can get your token from https://dashboard.ngrok.com/get-started/your-authtoken")
            return None

        # Start ngrok with explicit configuration
        ngrok_process = subprocess.Popen(
            ['ngrok', 'http', '--log=stdout', '5000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for ngrok to be ready
        print("Waiting for ngrok to start...")
        tunnels = wait_for_ngrok()
        
        if tunnels and tunnels.get('tunnels'):
            ngrok_url = tunnels['tunnels'][0]['public_url']
            print(f"\nNgrok URL: {ngrok_url}")
            print("Access this URL in your browser to test the IP detection")
            return ngrok_url
        else:
            print("Failed to get ngrok URL. Debug information:")
            print("1. Checking ngrok process...")
            os.system('ps aux | grep ngrok')
            print("\n2. Checking ngrok logs...")
            if ngrok_process.stdout:
                print(ngrok_process.stdout.read())
            if ngrok_process.stderr:
                print(ngrok_process.stderr.read())
            return None
            
    except FileNotFoundError:
        print("Error: ngrok not found. Please install ngrok first.")
        print("You can install it using: sudo snap install ngrok")
        return None
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        return None

if __name__ == '__main__':
    # Start ngrok
    ngrok_url = run_ngrok()
    
    if ngrok_url:
        # Open the ngrok URL in the default browser
        webbrowser.open(ngrok_url)
    else:
        print("\nContinuing without ngrok...")
        print("The application will still run, but IP detection might not work correctly.")
    
    # Run the Flask app
    print("\nStarting Flask application...")
    app.run(debug=True) 