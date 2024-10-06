import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Access the API key
AEROBOTICS_DEV_API_KEY = os.getenv('AEROBOTICS_DEV_API_KEY')
# Base URL for Aerobotics API
API_BASE_URL = "https://api.aerobotics.com/farming"
# Headers Used
headers = {
    'Authorization': f'Bearer {AEROBOTICS_DEV_API_KEY}',
    'Accept': 'application/json'
    }
