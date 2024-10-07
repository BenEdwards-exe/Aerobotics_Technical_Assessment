from flask import Flask, jsonify, request

import sys, os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Access the API key
MISSING_TREES_API_KEY = os.getenv('MISSING_TREES_API_KEY')


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orchard'))
from aerobotics_requests import *
from find_missing_trees import *

app = Flask(__name__)


@app.route("/", methods=['GET'])
def head():
    return "", 200


# HOSTNAME=”aero-test.my-test-site.com”
# curl \
# -X GET \
# -H ‘content-type:application/json’ \

# https://${HOSTNAME}/orchards/216269/missing-trees
@app.route("/orchards/<int:orchard_id>/missing-trees", methods=['GET'])
def get_missing_trees(orchard_id):
    """
    Run the missing trees finder from /orchard

    orchard_id : int
        Numerical ID of the orchard.
    """


    if type(orchard_id) != int:
        return jsonify({"message": "ERROR - Invalid argument type for orchard_id."}), 422


    headers = request.headers

    # content_type = headers.get('Content-Type')
    if request.content_type != 'application/json':
        return jsonify({'message': 'ERROR - Unsupported Media Type: Content-Type should be application/json'}), 415
   
   
    auth = headers.get("API-KEY")
    if auth != MISSING_TREES_API_KEY:
        return jsonify({"message": "ERROR - Unauthorized"}), 401
    


    # Details of orchard; for polygon 
    single_orchard_response = fetch_single_orchard(orchard_id)
    if single_orchard_response.status_code != 200:
        return single_orchard_response.json(), single_orchard_response.status_code


    # Get teh details of all surveys for current orchard
    survey_response = fetch_all_orchard_surveys(orchard_id)
    if survey_response.status_code != 200:
        return survey_response.json(), survey_response.status_code
    
    # Find the ID of the latest survey
    surveys = survey_response.json()["results"]  
    surveys_sorted_by_date = sorted(surveys, key=lambda survey: survey["date"])
    latest_survey_id = surveys_sorted_by_date[-1]["id"]

    # Get the trees for the latest survey
    tree_surveys_response = fetch_single_survey(latest_survey_id)
    if tree_surveys_response.status_code != 200:
        return tree_surveys_response.json(), survey_response.status_code
    

    # Run find missing trees method
    m = find_all_missing_trees(orchard_response_json=single_orchard_response.json(), survey_response_json=tree_surveys_response.json())
    m_resp = {"missing_trees" : m}
    return jsonify(m_resp), 200




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
