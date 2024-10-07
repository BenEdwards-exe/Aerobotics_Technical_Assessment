"""
This module is used to find the missing trees within an orchard, given the coordinates of the current trees
and the coordinates of the polygon around the orchard.
"""

import math
import geopy.distance
import numpy as np
from itertools import combinations
import geopandas as gpd
import pandas as pd
from shapely import LineString, Point, Polygon
from sklearn import metrics
from sklearn.cluster import KMeans


# Top Function
def find_all_missing_trees(orchard_response_json, survey_response_json) -> list[dict]:
    """
    Top function of module.
    Find all the missing trees for a given orchard.

    Parameters
    -----------
        orchard_response_json : Dict
            response.json() from GET https://api.aerobotics.com/farming/orchards/{orchard_id}/
        survey_response_json : Dict
            response.json() from GET https://api.aerobotics.com/farming/surveys/{survey_id}/tree_surveys/


    Returns
    -------------
        missing_points : List[Dict]
            [{ "lat": float, "lng": float }]\n
            List of dictionaries. Each containing the longitude and latitude for a missing tree.

    """
    
    # Extract polygon corners lng and lat coordinates
    polygon_latlongs = orchard_response_json["polygon"]
    polygon_lng_data = []
    polygon_lat_data = []
    for l in polygon_latlongs.split(" "):
        l_split = l.split(",")
        polygon_lng_data.append(float(l_split[0]))
        polygon_lat_data.append(float(l_split[1]))

    # Get the bounding box of the orchard polygon. 
    # This is just to have bounds when creating linestring geometries later
    polygon_points = [Point(xy) for xy in zip(polygon_lng_data, polygon_lat_data)]
    polygon_geometry = Polygon(polygon_points)
    poly_min_x, poly_min_y, poly_max_x, poly_max_y = polygon_geometry.bounds

    # Extract results from tree survey data and store in dataframe
    trees_df = pd.DataFrame(survey_response_json["results"])
    # X and Y coordinates of trees
    x = trees_df["lng"].to_numpy()
    y = trees_df["lat"].to_numpy()

    # Find the two direction slopes of the orchard
    dir_1_slope, dir_2_slope = find_orchard_slopes(x, y)

    # Cluster trees in each slope direction and get y-intercepts of each direction
    dir_1_y_intercepts, dir_1_labels = cluster_trees_by_slope(x, y, dir_1_slope)
    dir_2_y_intercepts, dir_2_labels = cluster_trees_by_slope(x, y, dir_2_slope)

    # Add the labels of the directional clustering to trees dataframe
    trees_df["dir_1_labels"] = dir_1_labels
    trees_df["dir_2_labels"] = dir_2_labels

    # Create geometry LineStrings for each row (cluster) in a direction
    dir_1_linestrings = linestrings_from_intercepts(dir_1_y_intercepts, dir_1_slope, poly_min_x, poly_max_x, poly_min_y, poly_max_y)
    dir_2_linestrings = linestrings_from_intercepts(dir_2_y_intercepts, dir_2_slope, poly_min_x, poly_max_x, poly_min_y, poly_max_y)

    # Calculate the distances between adjacent trees in a clustered direction
    dir_1_distances, dir_1_distances_dict = distance_between_adj_trees(trees_df, len(dir_1_y_intercepts), "dir_1_labels")
    dir_2_distances, dir_2_distances_dict = distance_between_adj_trees(trees_df, len(dir_2_y_intercepts), "dir_2_labels")

    # Get the IDs and distances of trees with outlier distances between them
    dir_1_outliers = find_distance_outliers(dir_1_distances, dir_1_distances_dict)
    dir_2_outliers = find_distance_outliers(dir_2_distances, dir_2_distances_dict)

    # Find all LineString intersections that are between trees that have been flagged as outliers
    dir_1_missing_points = find_intersects_between_trees(trees_df, dir_1_outliers, dir_1_linestrings, dir_2_linestrings)
    dir_2_missing_points = find_intersects_between_trees(trees_df, dir_2_outliers, dir_2_linestrings, dir_1_linestrings)

    # Remove all duplicate points
    missing_points_combined = list(set(dir_1_missing_points + dir_2_missing_points))

    # Missing Trees to dict output format
    missing_trees = []
    for point in missing_points_combined:
        missing_trees.append({"lat": point.y, "lng": point.x})

    return missing_trees
    


def find_orchard_slopes(x_coords, y_coords) -> tuple[float, float]:
    """
    Calcualte the two directional slopes that orchard trees are planted in.

    Parameters
    -------------
        x : np.array
            Numpy array of tree longitudes (x-coords).
        y : np.array
            Numpy array of tree latitudes (y-coords).

    Returns
    --------------
        (slope_1, slope_2) : (float, float)
            Two slopes.
    """

    # Index combinations used to form vectors. Index of tree from and tree to.
    idx_combinations = np.array(list(combinations(range(len(x_coords)),2)))

    # Find horizontal and vertical deltas (difference between tree coordinates) between combinations.
    delta_x = x_coords[idx_combinations[:, 0]] - x_coords[idx_combinations[:, 1]]
    delta_y = y_coords[idx_combinations[:, 0]] - y_coords[idx_combinations[:, 1]]

    # Masks for selecting direction vecotrs from deltas.
    # For an orchard with two slopes, one slope's gradient will have a larger x-delta than its y-delta,
    # while the other slope will have a larger y-delta than its x-delta. (except for a perfect 45-degree orthogonal cross)
    dir_1_mask = np.abs(delta_x) < np.abs(delta_y)
    dir_2_mask = np.abs(delta_x) > np.abs(delta_y)

    slopes = [] # list to store the two slopes

    for mask in [dir_1_mask, dir_2_mask]:
        delta_x_masked = delta_x[mask]
        delta_y_masked = delta_y[mask]

        # Distance between trees - magnitude of vector deltas
        magnitude = np.sqrt(np.square(delta_x_masked) + np.square(delta_y_masked))

        # Sort magnitude of difference and select first 20 percent (of smallest vectors)
        # Smaller magnitudes should be trees close to each other in the same row
        idx_of_sorted = np.argsort(magnitude)
        idx_to_select = idx_of_sorted[:int(len(x_coords)*0.2)]

        delta_x_sampled = delta_x_masked[idx_to_select]
        delta_y_sampled = delta_y_masked[idx_to_select]

        # Reverse vectors with negative x-components (to make sure we move in one direction)
        idx_to_reverse = delta_x_sampled < 0.0
        delta_x_sampled[idx_to_reverse] = -1.0 * delta_x_sampled[idx_to_reverse]
        delta_y_sampled[idx_to_reverse] = -1.0 * delta_y_sampled[idx_to_reverse]

        # Calcualte the slope of a direction by taking the mean deltas in a direction
        dir_slope = np.mean(delta_y_sampled) / np.mean(delta_x_sampled)

        # Append to slopes list
        slopes.append(dir_slope)

    # convert list to tupe and return
    return tuple(slopes)




def cluster_trees_by_slope(x_coords , y_coords, slope_of_direction, min_clusters=2, max_clusters=50) -> tuple[np.array, np.array]:
# Adapted from https://stackoverflow.com/questions/75208601/how-can-i-cluster-coordinate-values-into-rows-using-their-y-axis-value 
# by using the manhattan distance instead of Euclidian.
    """
    Cluster the trees into rows for a specified slope based on their coordinates. This method iteratively runs k-means for 
    clusters between min_clusters and max_clusters. The best fitting model is then selected.

    Parameters
    -------------
        x_coords : numpy.array
            Numpy array of tree longitudes.
        y_coords : numpy.array
            Numpy array of tree latitudes.
        slope_of_direction : float
            Slope of the direction that trees should be clustered in. Each orchard has
            two directions.
        min_clusters : int
            Minimum clusters to search from for K-Means.
        max_clusters : int
            Maximum clusters to search from for K-Means.

    Returns
    -----------
        (y_intercepts, y_labels) : (np.array, np.array)
            y_intercepts: Y-axis (latitude value) intercepts of the cluster row.\n
            y_labels: Cluster labels mapped to the array of y coordinates.
    """

    # Clustering is done on the vertical distance of the line that passes the origin -> y_origin
    # Calcluate y_origin from the slope of the direction and the x_coordinate
    y_origin = y_coords - slope_of_direction*x_coords

    # y_origin is a 1D array. Need to convert it to a 2D array.
    y_origin_reshaped = y_origin.reshape(-1,1)

    max_score_iter1 = -math.inf # maximum score reached in the first K-Means search
    best_k_iter1 = 0 # number of classes that results in the best fitting K-Means model
    for k in range(min_clusters, max_clusters):
        model = KMeans(n_clusters=k)
        model_res = model.fit(y_origin_reshaped)
        # Using manhattan metric instead of euclidean 
        # TODO: Silhoute score distance metric to be modified with orthogonal score (Possible Improvement)
        score = metrics.silhouette_score(y_origin_reshaped, model_res.labels_.astype(float), metric='manhattan')
        if (score > max_score_iter1): # Find the best fitting model based on the silhoutte score
            max_score_iter1 = score
            best_k_iter1 = k
    
    # The first iterative search of K-Means is sometimes out by one cluster.
    # Run the search again, but now only evaluate for clusters in the range [best_k_iter1 - 1, best_k_iter1 + 1]
    final_model = KMeans() # the best model from the second search
    max_score_iter2 = -math.inf
    for i in range(3): 
        for k in range(best_k_iter1-1, best_k_iter1+2):
            model = KMeans(n_clusters=k)
            model_res = model.fit(y_origin_reshaped)
            # Using manhattan metric instead of euclidian
            score = metrics.silhouette_score(y_origin_reshaped, model_res.labels_.astype(float), metric='manhattan')
            if (score > max_score_iter2): # Store the highest scoring model
                max_score_iter2 = score
                final_model = model_res
    

    # The y-axis intercepts for each cluster
    y_intercepts = np.squeeze(final_model.cluster_centers_, axis=1) 

    # Cluster label for each tree
    y_labels = np.array(final_model.labels_)

    return y_intercepts, y_labels


def linestrings_from_intercepts(y_intercepts, dir_slope, x_min, x_max, y_min, y_max) -> list[LineString]:
    """
    Create Shapely geometry LineStrings based on the intercepts and slope for a direction.
    Each LineString is the centreline of a cluster in one of the two directions.
    Bound the LineString ends in the bouding box of the orchard.

    Parameters
    ----------
        y_intercepts : numpy.array
            Numpy array of y-intercepts.
        dir_slope : float
           Slope for current direction.
        x_min, x_max, y_min, y_max : float
            Bounding box of orchard polygon.

    Returns
    --------------
        direction_linestrings : [LineString]
            List of LineStrings for current direction
    """


    linestring_points = [] # list of points used to form LineStrings

    # Calculate the two points required to form a LineString
    for i in range(len(y_intercepts)):
        # depending on the gradient of the slope, a LineString is bounded by the top and bottom OR left and right bounds of the orchard polygon
        if (dir_slope > 0): 
            point_1 = ((y_min - y_intercepts[i])/dir_slope, y_min)
            point_2 = ((y_max - y_intercepts[i])/dir_slope, y_max)
            linestring_points.append([point_1,point_2])
        else:
            point_1 = (x_min, x_min*dir_slope + y_intercepts[i])
            point_2 = (x_max, x_max*dir_slope + y_intercepts[i])
            linestring_points.append([point_1,point_2])

    # Create LineStrings for the current direction
    dir_linestrings = [LineString(points) for points in linestring_points]
    
    return dir_linestrings            


def distance_from_origin(row):
    """Helper function to calculate distance from origin when sorting the trees DataFrame"""
    return np.sqrt(row['lng']**2 + row['lat']**2)


def distance_between_adj_trees(tree_classed_df, n_labels, dir_col_name: str):
    """
    Find the distance between two adjacent trees, that have been clustered in a certain direction. 

    Parameters
    --------------
        tree_classed_df : DataFrame
            DataFrame of orchard trees with classes in the two directions.
        n_labels : int
            Amount of clusters for current direction.
        dir_col_name : str
            \'dir_1_labels\' or \'dir_2_labels\'

    Returns
    --------------
        (all_distances_combined, dir_distances_dict) : (List, Dict)
            all_distances_combined: List of all distances in the direction. This is used to calculate the mean and standard deviation in a direction.\n
            dir_distances_dict: Dictionary of the distances for each label. dir_distances_dict[label] = (direction_label, tree_from_id, tree_to_id, distance_between)
    """


    dir_distances_dict = {} # dict to store distance between trees respective of their direction label and IDs
    all_distances_combined = [] # list to store all distances, independent of label

    for label in range(n_labels):
        # Get all trees in the direction of direction label
        dir_label_trees_df = tree_classed_df[tree_classed_df[dir_col_name].values == label]
        
        # Sort trees in the direction
        # Because of the variance in x-coordinates, trees in a direction can be sorted incorrectly if they are sorted from left to right (same for bottom to top).
        # Distance from origin is used for sorting.
        dir_label_trees_df = dir_label_trees_df.assign(distance=dir_label_trees_df.apply(distance_from_origin, axis=1)).sort_values(by='distance')
        
        # DataFrame columns to numpy arrays
        lngs = dir_label_trees_df["lng"].to_numpy()
        lats = dir_label_trees_df["lat"].to_numpy() 
        ids = dir_label_trees_df["id"].to_numpy()
        dir_labels = dir_label_trees_df[dir_col_name].to_numpy()

        # Store the distances between trees in a specific direction
        distance_data = [] # [(direction_label, tree_from_id, tree_to_id, distance_between),...]
            
        # Get the geographic distance between two sequential trees in the current direction of travel
        for i in range(0, len(lngs)-1):

            direction_label = dir_labels[i]
            tree_from_id = ids[i]
            tree_to_id = ids[i+1]

            coords_1 = (lats[i], lngs[i])
            coords_2 = (lats[i+1], lngs[i+1])
            # geographic distance between trees in meters
            distance_between = geopy.distance.geodesic(coords_1, coords_2).m

            all_distances_combined.append(distance_between) # used to calcualte mean and std of a direction.
            distance_data.append((direction_label, tree_from_id, tree_to_id, distance_between))
        
        dir_distances_dict[label] = distance_data

    return all_distances_combined, dir_distances_dict




def find_distance_outliers(distances_all, distances_dict):
    """
    Return all the outlier distances with the IDs for the tree from and tree to.
    Outliers have a z-score of more than 3.

    Parameters
    ------------------------------
        distances_all : List
            List of all distances in the direction.
        distances_dict : Dict
            Dictionary of the distances for each label. dir_distances_dict[label] = (label, tree_from_id, tree_to_id, distance_between)


    Returns
    ------------------------
        List (dir_label, tree_from_id, tree_to_id, distance) : List[(int,int,int,float)]
            All the outliers.
    """
    
    # Meand and Std for current direction
    mean = np.mean(distances_all)
    std = np.std(distances_all)
 
    outliers = [] # list to store the IDs and distance of outlier distances.
    z_threshold = 3

    # loop through all the distances between adjacent trees in a direction
    for dir_label in distances_dict: 
        dir_dists = distances_dict[dir_label]
        for dist_data in dir_dists:
            z_score = (dist_data[3] - mean)/std
            if ( abs(z_score) > z_threshold ):
                outliers.append(dist_data)

    return outliers



def find_intersects_between_trees(trees_classed_df, dir_outliers, curr_dir_linestrings, other_dir_linestrings) -> list[Point]:
    """
    Find all grid intersections on LineStrings with trees that have outlier distances between them.

    Parameters
    ---------------
        trees_classed_df : DataFrame
            DataFrame of trees with class labels.
        dir_outliers : Dict
            Dict containing the IDs of trees with outlier distances between them.
            dir_outliers[label] = (outlier_dir_label, tree_from_id, tree_to_id, distance_between)
        curr_dir_linestrings : List[LineString]
            List of LineStrings in the direction of consideration.
        other_dir_linestrings : List[LineString]
            List of LineStrings in the other direction.

    Returns
    --------------
        missing_points : List[Point]
            List of Shapely Points where missing trees are.
    """
    

    missing_points = []

    # Loop through all outliers in the current direction
    for n in range(len(dir_outliers)):
        # Get the linestring that the current outlier is on
        outlier_dir_label = dir_outliers[n][0]
        outlier_line = curr_dir_linestrings[outlier_dir_label]

        intersection_points = []

        # Find all points of intersection between outlier_line and all linestrings of the other direction
        for l in other_dir_linestrings:
            intersections = outlier_line.intersection(l)
            if not intersections.is_empty:
                intersection_points.append(intersections)

        # Sort the intersection points by their horizontal coordinate
        intersection_points_sorted = sorted(intersection_points, key=lambda p: p.x)

        # Create Shapely Point for tree_from and tree_to
        tree_from_id = dir_outliers[n][1]
        tree_to_id = dir_outliers[n][2]

        tree_from_mask = trees_classed_df['id'] == tree_from_id
        tree_to_mask = trees_classed_df['id'] == tree_to_id

        tree_from_lng = trees_classed_df.loc[tree_from_mask, 'lng'].iloc[0]
        tree_from_lat = trees_classed_df.loc[tree_from_mask, 'lat'].iloc[0]

        tree_to_lng = trees_classed_df.loc[tree_to_mask, 'lng'].iloc[0]
        tree_to_lat = trees_classed_df.loc[tree_to_mask, 'lat'].iloc[0]

        tree_from_point = Point(tree_from_lng, tree_from_lat)
        tree_to_point = Point(tree_to_lng, tree_to_lat)

        # Find the index of point for tree_from and the index of point for tree_to from the intersection points
        tree_from_idx = min(range(len(intersection_points_sorted)), key=lambda i: intersection_points_sorted[i].distance(tree_from_point))
        tree_to_idx = min(range(len(intersection_points_sorted)), key=lambda i: intersection_points_sorted[i].distance(tree_to_point))

        # Swap if upper index is smaller (i.e., idx from tree 5 to tree 2)
        if (tree_from_idx > tree_to_idx): 
            temp = tree_from_idx
            tree_from_idx = tree_to_idx
            tree_to_idx = temp

        # Find all the intersection points between tree_from and tree_to and append to missing_points.
        # This is for when two or more sequential trees are missing in a direction.
        for i in range(tree_from_idx+1, tree_to_idx):
            missing_points.append(intersection_points_sorted[i])

    
    return missing_points       




if __name__ == "__main__":
    pass