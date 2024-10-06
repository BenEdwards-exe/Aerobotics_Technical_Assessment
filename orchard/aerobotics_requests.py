from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()
# Access the API key
AEROBOTICS_DEV_API_KEY = os.getenv('AEROBOTICS_DEV_API_KEY')
API_BASE_URL = "https://api.aerobotics.com/farming"






if __name__ == "__main__":
    # print(f"Your API key is: {AEROBOTICS_DEV_API_KEY}")
    print(AEROBOTICS_DEV_API_KEY)
    headers = {
    'Authorization': f'Bearer {AEROBOTICS_DEV_API_KEY}',
    'Accept': 'application/json'
    }
    print(headers)
    orchard_id = 216269
    survey_response = requests.request(
    "GET", f"{API_BASE_URL}/surveys/?orchard_id={orchard_id}", headers=headers)
    print(survey_response.json())