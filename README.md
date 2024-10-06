# Aerobotics Technical Assignment

Technical Assessment for Software Engineer (Machine Learning) at Aerobotics

## Assumption(s)
1. Trees are planted in a grid with two main directions.
2. The distance between trees in a certain direction (on a line) is the same (with some variance).
3. A gap of **more than one tree** in a specific direction can exist.
4. Missing trees on corners are ignored.

## Method for finding missing trees

1. For each of the two directions, trees are clustered in rows using K-Means.
2. Gridlines are formed for orchard.
3. The distance/gap between adjacent trees in a cluster (direction) is calculated.
4. Gaps between trees with a z-score of more than 3 are identified as outlier gaps.
5. For each outlier gap in a cluster, the trees on opposite ends of the gap are mapped to the intersections of the gridlines from step 2. The grid intersections between these trees are stored as the locations of missing trees.
6. Duplicate points are removed from the list of locations.

## Visual Example
A visual step-by-step example of how missing trees are found is provided in [find_missing_visualised.ipynb](https://github.com/BenEdwards-exe/Aerobotics_Technical_Assessment/blob/main/find_missing_visualised.ipynb). For testing, there is also the option to randomly remove a percentage of trees from the orchard. For 5% removed, it still seems to work well.

## API Reference
### **GET** /orchards/{orchard_id}/missing-trees

**Request Headers**
| Key | Type | Value |
| ----------- | ----------- | ----------- |
| API-KEY | string | Your unique key for missing trees |
| content-type | string | application/json | 

**Example Request:**
> curl -X GET "http://\${HOSTNAME}/orchards/216269/missing-trees" -H "API-KEY: ${YOUR_API_KEY}" -H "content-type: application/json"


**Example Response:**
```
{
    "missing_trees": [
        { "lat": -32.32862520358892, "lng": 18.825669547116767 },
        { "lat": -32.32890469467948, "lng": 18.825865655754654 },
        { "lat": -32.32880084663632, "lng": 18.826435883251875 },
        { "lat": -32.32868098269371, "lng": 18.826676939956673 }
    ]
}
```


## Docker Build
> docker-compose up --build



## Possible Improvements to Investigate
- When the trees are clustered in a particular direction, the optimal number of clusters is determined with the [silhoutte score](https://scikit-learn.org/1.5/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) using the manhattan distance. Silhoutte score uses the mean of inter-cluster distances and intra-cluster distances. However, opposed to normal clustering, the centroids of orchard rows can actually be seen as the entire line of the row. Therefore, a possible improvement could be to rather use the orthogonal distance between the point of a cluster and the line of the row.

- Faster convergence of K-Means. This is the largest bottleneck.

- For more than 5% removed, I think not all the outliers are identified. Maybe try an iterative approach by running twice or by lowering the z-threshold.




