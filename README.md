# CAPTCHA Bypassing Project

This project demonstrates CAPTCHA bypassing techniques using machine learning models. It includes a Flask web application that generates and processes CAPTCHAs, with IP detection capabilities through ngrok integration.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ngrok (for IP detection testing)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ct101apcs/CAPTCHA-bypassing.git
cd CAPTCHA-bypassing
```

### 2. Set Up Python Virtual Environment

#### Ubuntu/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install ngrok

#### Ubuntu/Linux:
```bash
# Using snap (recommended)
sudo snap install ngrok

# Or using direct download
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin
```

#### Windows:
1. Download ngrok from https://ngrok.com/download
2. Extract the zip file
3. Move ngrok.exe to a directory in your PATH or use the full path

#### macOS:
```bash
# Using Homebrew
brew install ngrok

# Or using direct download
brew install wget
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip
unzip ngrok-v3-stable-darwin-amd64.zip
sudo mv ngrok /usr/local/bin
```

#### GitHub Codespaces:
```bash
# Install ngrok
curl -O https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin
```

### 4. Configure ngrok

1. Sign up for a free ngrok account at https://dashboard.ngrok.com/signup
2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
3. Authenticate ngrok:
```bash
ngrok authtoken YOUR_TOKEN_HERE
```

## Running the Application

### If you need to work with the IP detection (run with ngrok):

The helper script will automatically start both ngrok and the Flask application:

```bash
python run_with_ngrok.py
```

This will:
1. Start ngrok and create a public URL
2. Open the URL in your default browser
3. Start the Flask application

### If you don't need ngrok:

Run the Flask application directly:
```bash
flask run
```

3. Access the application through the ngrok URL shown in the ngrok terminal

## Features

- CAPTCHA generation and processing
- Multiple transformation options
- IP detection through ngrok
- Real-time model predictions
- Session management
- Access logging

## Project Structure

```
CAPTCHA-bypassing/
├── Datasets/                 # Completely similar to the original repo
   ├── archive/            
      ├── animals/
         ├── animals/
   ├── animal_dataset.py
   ├── classes.py
├── captcha_generator.py      # CAPTCHA generation logic
├── model_predictionss.py     # Model prediction logic
├── app.py                    # Main Flask application
├── run_with_ngrok.py         # Helper script for ngrok integration
├── requirements.txt          # Python dependencies
├── access_logs/              # Directory for IP access logs
└── templates/                # HTML templates
```

## Troubleshooting

### Common Issues

1. **ngrok not found**
   - Ensure ngrok is installed and in your PATH
   - Try using the full path to ngrok

2. **Authentication Error**
   - Make sure you've run `ngrok authtoken YOUR_TOKEN_HERE`
   - Check if your token is valid

3. **Port Already in Use**
   - Change the port in both ngrok and Flask commands
   - Kill any existing ngrok processes: `pkill ngrok`

4. **Connection Refused**
   - Ensure Flask is running before starting ngrok
   - Check if the port matches in both applications

### Debugging

- Check ngrok status: `ngrok config check`
- View ngrok logs: `ngrok http --log=stdout 5000`
- Check access logs in `access_logs/access_logs.json`