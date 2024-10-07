from flask import Flask, jsonify, request
import sys, os

# Append orchard to sys path to import modules from ../orchard
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orchard'))
from aerobotics_requests import *
from find_missing_trees import *

from dotenv import load_dotenv


# Load variables from .env file
load_dotenv()
# Access the API key
MISSING_TREES_API_KEY = os.getenv('MISSING_TREES_API_KEY')

# Create Flask WSGI application object
app = Flask(__name__)


@app.route("/", methods=['GET'])
def get_base_url():
    """
    Return an empty HTTP OK response when the base url of the API is accessed. 
    This is to prevent Render.com hosting from entering a 404 loop.
    """
    return "", 200


@app.route("/orchards/<int:orchard_id>/missing-trees", methods=['GET'])
def get_missing_trees(orchard_id):
    """
    Return the missing trees from a specific orchard.\n
    Calls the methods from \'find_missing_trees.py\'
    
    Args
    ----------
        orchard_id : int
            The numerical ID of the orchard.

    Returns
    ----------
        response : json
            A JSON object containing a list of the missing tree latitudes and longitudes.
            An HTTP error code is with a message is returned if either of the following is is supplied: 
            an incorrect argument type, unsupported media type, the incorrect API Key, or an orchard ID
            that does not exist in the Aerobotics Database.

        Example:
        {"missing_trees": [{ "lat": float, "lng": float }]}, 200

    """

    # Check if supplied orchard_id is an integer.
    if type(orchard_id) != int:
        return jsonify({"message": "ERROR - Invalid argument type for orchard_id."}), 422

    # Extract the headers from the HTTP request.
    headers = request.headers

    # Check if the header content type is JSON format.
    if request.content_type != 'application/json':
        return jsonify({'message': 'ERROR - Unsupported Media Type: Content-Type should be application/json'}), 415
   
    # Check if the correct API Key is provided
    auth = headers.get("API-KEY")
    if auth != MISSING_TREES_API_KEY:
        return jsonify({"message": "ERROR - Unauthorized"}), 401
    

    # --- If all of the obove checks are passed, then the information about the requested orchard is queried from
    # --- the Aerobotics database. For each Aerobotics query, a check is run to see if the HTTP 200 code was received.

    # Fetch the details of a single orchard.
    # This is used to construct the polygon around the trees.
    single_orchard_response = fetch_single_orchard(orchard_id)
    if single_orchard_response.status_code != 200:
        return single_orchard_response.json(), single_orchard_response.status_code

    # Fetch the details of all surveys conducted for the orchard.
    survey_response = fetch_all_orchard_surveys(orchard_id)
    if survey_response.status_code != 200:
        return survey_response.json(), survey_response.status_code
    
    # Sort the surveys to find the ID of the most most recent survey.
    surveys = survey_response.json()["results"]  
    surveys_sorted_by_date = sorted(surveys, key=lambda survey: survey["date"])
    latest_survey_id = surveys_sorted_by_date[-1]["id"]

    # Fetch the details of the most recent orchard survey.
    tree_surveys_response = fetch_single_survey(latest_survey_id)
    if tree_surveys_response.status_code != 200:
        return tree_surveys_response.json(), survey_response.status_code
    

    # Run the top method of find_missing_trees.py
    # This returns a list of the latitude and longitude coordinates of the missing trees.
    missing_tree_latlongs = find_all_missing_trees(orchard_response_json=single_orchard_response.json(), survey_response_json=tree_surveys_response.json())
    response_dict = {"missing_trees" : missing_tree_latlongs}

    # Return response with missing trees and HTTP OK
    return jsonify(response_dict), 200



if __name__ == "__main__":
    # Run the Flask API in debug mode
    app.run(debug=True, host='0.0.0.0')
