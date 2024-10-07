import requests

# Import Global Module Variables from Config fyle
from config import headers
from config import API_BASE_URL

# Fetch the details of a single orchard
def fetch_single_orchard(orchard_id: int) -> requests.Response:
    """
    Fetch the details of a single orchard from the Aerobotics Developer API.

    Args
    -------
    orchard_id : int
        Numerical ID of the orchard.
    """
    url =  f"{API_BASE_URL}/orchards/{orchard_id}/"
    res = requests.get(url, headers=headers)
    return res

# Fetch data from all orchard surveys
def fetch_all_orchard_surveys(orchard_id: int) -> requests.Response:
    """
    Fetch the details of all the surveys conducted for an 
    orchard from the Aerobotics Developer API.

    Args
    ---------
    orchard_id : int
        Numerical ID of the orchard.
    """
    url = f"{API_BASE_URL}/surveys/?orchard_id={orchard_id}"
    res = requests.get(url, headers=headers)
    return res


# Fetch data for single orchard survey
def fetch_single_survey(survey_id: int) -> requests.Response:
    """
    Fetch the details of a single survey conducted for an 
    orchard from the Aerobotics Developer API.

    Args
    -----------
    orchard_id : int
        Numerical ID of the orchard.
    """
    url = f"{API_BASE_URL}/surveys/{survey_id}/tree_surveys/"
    res = requests.get(url, headers=headers)
    return res



if __name__ == "__main__":
    pass