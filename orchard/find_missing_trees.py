import math
import geopy.distance
import numpy as np
from itertools import combinations
import geopandas as gpd
import pandas as pd
from shapely import LineString, Point, Polygon
from sklearn import metrics
from sklearn.cluster import KMeans


# TODO: Function description
def cluster_trees_by_slope(x,y,dir_slope,min_clusters=2,max_clusters=50) -> tuple[np.array, np.array]:
# Adapted from https://stackoverflow.com/questions/75208601/how-can-i-cluster-coordinate-values-into-rows-using-their-y-axis-value by using the manhattan distance instead of Euclidian.
# TODO: Silhoute score distance metric to be modified with orthogonal score

    y_adj = y - dir_slope*x
    y_adj_reshaped = y_adj.reshape(-1,1)

    max_score = -math.inf
    best_k_1 = 0
    for k in range(min_clusters, max_clusters):
        model = KMeans(n_clusters=k)
        model_res = model.fit(y_adj_reshaped)
        # Using manhattan metric instead of euclidean 
        score = metrics.silhouette_score(y_adj_reshaped, model_res.labels_.astype(float), metric='manhattan')
        if (score > max_score):
            max_score = score
            best_k_1 = k
    
    # Since the cluster estimate is sometimes out by one 'row' in current direction,
    # run again for best_k_1-1 to best_k_1+1
    final_model_res = KMeans()
    max_score_2 = -math.inf
    for i in range(3):
        for k in range(best_k_1-1, best_k_1+2):
            model = KMeans(n_clusters=k)
            model_res = model.fit(y_adj_reshaped)
            # Using manhattan metric instead of euclidian
            score = metrics.silhouette_score(y_adj_reshaped, model_res.labels_.astype(float), metric='manhattan')
            if (score > max_score_2):
                max_score_2 = score
                final_model_res = model_res
    

    # # Look for most common best_k_2 found
    # most_common_k = np.bincount(np.array(best_k_log)).argmax()

    # # Final fit for the best model
    # final_model = KMeans(n_clusters=most_common_k)
    # final_model_res = final_model.fit(y_adj_reshaped)

    # The y intercepts for y_adj (Where the line fitted to each 'row' crosses the y-axis)
    y_intercepts = np.squeeze(final_model_res.cluster_centers_, axis=1) 

    # Tree labels
    y_labels = np.array(final_model_res.labels_)

    return y_intercepts, y_labels

# TODO: Funciton description
def find_orchard_slopes(x,y):
    assert(len(x) == len(y))
    # Index combinations used to form vectors
    idx_combinations = np.array(list(combinations(range(len(x)),2)))

    # Find horizontal and vertical deltas between combinations
    delta_x = x[idx_combinations[:, 0]] - x[idx_combinations[:, 1]]
    delta_y = y[idx_combinations[:, 0]] - y[idx_combinations[:, 1]]

    # We have two directions that straight rows can be in
    # dir_1: delta_y is larger than delta_x
    # dir_2: delta_x is larger than delta_y

    # Masks for selecting direction vecotrs from deltas
    dir_1_mask = np.abs(delta_x) < np.abs(delta_y)
    dir_2_mask = np.abs(delta_x) > np.abs(delta_y)

    slopes = []

    for mask in [dir_1_mask, dir_2_mask]:
        delta_x_masked = delta_x[mask]
        delta_y_masked = delta_y[mask]

        delta_mag = np.sqrt(np.square(delta_x_masked) + np.square(delta_y_masked))
        # Sort magnitude of difference and select/sample first 20 percent
        idx_of_sorted = np.argsort(delta_mag)
        idx_to_select = idx_of_sorted[:int(len(x)*0.2)]

        delta_x_sampled = delta_x_masked[idx_to_select]
        delta_y_sampled = delta_y_masked[idx_to_select]

        # Reverse vectors with negative x-xomponents
        idx_to_reverse = delta_x_sampled < 0.0
        delta_x_sampled[idx_to_reverse] = -1.0 * delta_x_sampled[idx_to_reverse]
        delta_y_sampled[idx_to_reverse] = -1.0 * delta_y_sampled[idx_to_reverse]

        # Calcualte the mean slope of direction
        dir_slope = np.mean(delta_y_sampled) / np.mean(delta_x_sampled)
        # Append to output
        slopes.append(dir_slope)

    return tuple(slopes)

# TODO: Funciton description
def linestrings_from_intercepts(y_intercepts, dir_slope, x_min, x_max, y_min, y_max) -> list[LineString]:

    n = len(y_intercepts)
    linestring_points = []

    for i in range(n):
        if (dir_slope > 0):
            p_1 = ((y_min - y_intercepts[i])/dir_slope, y_min)
            p_2 = ((y_max - y_intercepts[i])/dir_slope, y_max)
            linestring_points.append([p_1,p_2])
        else:
            p_1 = (x_min, x_min*dir_slope + y_intercepts[i])
            p_2 = (x_max, x_max*dir_slope + y_intercepts[i])
            linestring_points.append([p_1,p_2])

    dir_linestrings = [LineString(points) for points in linestring_points]
    
    return dir_linestrings            

# TODO: Funciton description
def distance_from_origin(row):
    return np.sqrt(row['lng']**2 + row['lat']**2)

# Find the distance between two adjacent trees in a certain direction
# TODO: Funciton description
def distance_between_adj_trees(tree_classed_df, n_labels, dir_col_name: str):
    dir_distances_dict = {} # dict to store distance between trees respective of their direction label and ids
    dir_all_distances = [] # list to store all distances, independent of label

    for label in range(n_labels):
        # Get all trees in the direction of direction label
        dir_label_trees_df = tree_classed_df[tree_classed_df[dir_col_name].values == label]
        
        # Sort trees in the direction
        dir_label_trees_df = dir_label_trees_df.assign(distance=dir_label_trees_df.apply(distance_from_origin, axis=1)).sort_values(by='distance')
        
        # Cols to numpy arrs
        lngs = dir_label_trees_df["lng"].to_numpy()
        lats = dir_label_trees_df["lat"].to_numpy() 
        ids = dir_label_trees_df["id"].to_numpy()
        dir_labels = dir_label_trees_df[dir_col_name].to_numpy()

        # Get the distance between two sequential trees in the current direction of travel
        distance_data = [] # [(direction_label, tree_from_id, tree_to_id, distance_between),...]
            
        for i in range(0,len(lngs)-1):

            curr_direct_label = dir_labels[i]
            tree_from_id = ids[i]
            tree_to_id = ids[i+1]

            coords_1 = (lats[i], lngs[i])
            coords_2 = (lats[i+1], lngs[i+1])
            distance_between = geopy.distance.geodesic(coords_1, coords_2).m

            dir_all_distances.append(distance_between)
            distance_data.append((curr_direct_label, tree_from_id, tree_to_id, distance_between))
        
        dir_distances_dict[label] = distance_data


    return dir_all_distances, dir_distances_dict

# Return outlier distances between trees
# Return List[(dir_label, tree_from_id, tree_to_id, distance),...]
# TODO: Funciton description
def find_distance_outliers(distances_all, distances_dict):
    # Meand and Std for current direction
    mean = np.mean(distances_all)
    std = np.std(distances_all)
 
    outliers = []
    z_threshold = 3
    
    for dir_label in distances_dict:
        dir_dists = distances_dict[dir_label]
        for dist_data in dir_dists:
            z_score = (dist_data[3] - mean)/std
            if ( abs(z_score) > z_threshold ):
                outliers.append(dist_data)

    return outliers

# Find all grid intersections between outlier distances
# TODO: Funciton description
def find_intersects_between_trees(trees_classed_df, dir_outliers, curr_dir_linestrings, other_dir_linestrings):
    missing_points = []

    # Loop through all outliers in the current direction
    for n in range(len(dir_outliers)):
        # Get the linestring that the current outlier is on
        outlier_dir_label = dir_outliers[n][0]
        outlier_line = curr_dir_linestrings[outlier_dir_label]

        intersection_points = []

        # Find all points of intersection between outlier_line and linestrings of the other direction
        for l in other_dir_linestrings:
            intersections = outlier_line.intersection(l)
            if not intersections.is_empty:
                intersection_points.append(intersections)

        # Sorth the intersection points by their horizontal coordinate
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

        # swap if upper index is smaller (i.e., idx from tree 5 to tree 2)
        if (tree_from_idx > tree_to_idx): 
            temp = tree_from_idx
            tree_from_idx = tree_to_idx
            tree_to_idx = temp

        # Find all the intersection points between tree_from and tree_to and append to missing_points
        # This is for when two or more sequential trees are missing in a direction
        for i in range(tree_from_idx+1, tree_to_idx):
            missing_points.append(intersection_points_sorted[i])
    
    return missing_points       


# orchard_response_json: GET https://api.aerobotics.com/farming/orchards/{orchard_id}/
# survey_response_json: GET https://api.aerobotics.com/farming/surveys/{survey_id}/tree_surveys/
# TODO: Funciton description
def find_all_missing_trees(orchard_response_json, survey_response_json) -> list[dict]:
    
    # Extract polygon corners lng and lat values
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
    # X and Y values of trees
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

    # Get the outlier distances and ids of trees involved
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
    


if __name__ == "__main__":
    pass