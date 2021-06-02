
# K-Means Clustering and Analyzing Random Forest Algorithms in Python with JupyterLab
### Predictive Analysis for Breast Cancer Diagnostics 

####  Source
This dataset was pulled from Kaggle, and comprises a set of nuclei features for breast tumor diagnosis. 

Source and feature details:
>[UCI ML Breast Cancer Wisconsin Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
>Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
>Attribute Information:

>1) ID number

>2) Diagnosis (M = malignant, B = benign)

>3-32)Ten real-valued features are computed for each cell nucleus:

>a) radius (mean of distances from center to points on the perimeter)

>b) texture (standard deviation of gray-scale values)

>c) perimeter

>d) area

>e) smoothness (local variation in radius lengths)

>f) compactness (perimeter^2 / area - 1.0)

>g) concavity (severity of concave portions of the contour)

>h) concave points (number of concave portions of the contour)

>i) symmetry

>j) fractal dimension ("coastline approximation" - 1)

>The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, >field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

####  Initial Cleaning and Processing Steps

I indexed the first column to avoid any errors, and converted the diagnosis from M/B to a binary operator. I split the diagnosis off into a separate 'y' dataframe, and pushed the predictors to an 'x' dataset. 

####  Principal Component Analysis

Here's the initial look at the variance explained by the 30 features:

| 1     | 2    | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    |
|-------|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 44.3  | 63.3 | 72.7  | 79.3  | 84.8  | 88.8  | 91.1  | 92.7  | 94.1  | 95.3  |

|  11    |   12   |  13    |  14    |  15    |  16    |  17    |  18    |  19    |  20    |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  96.3  |   97.2 |  98.0  |  98.5  |  98.8  |  99.1  |  99.3  |  99.5  |  99.7  |  99.8  |

|  21    | 22   | 23    | 24    | 25    | 26    | 27    | 28    | 29    | 30    |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|  99.8  |  99.9 | 100.0 | 100.1 | 100.2 | 100.3 | 100.3 | 100.3 | 100.3 | 100.3 |

This dataset is a great example of one that can be improved with Principal Component Analysis - we can see that the first 10 principal components account for over 95% of the variation. 

![PCA](https://github.com/ElishaPhillips/Python-K-Means-RandomForest-Wisconsin-Breast-Cancer-Diagnostics/blob/067a1fe05c20a5ec0574d580becd5664fd1c97c9/Graphs/pca.png)

Here's a quick graph of the first 15 components plotted against the last 15:
![PCA30](https://github.com/ElishaPhillips/Python-K-Means-RandomForest-Wisconsin-Breast-Cancer-Diagnostics/blob/9c7d169e99817ed944578b93f33bdc127881913a/Graphs/Visualising30.png)

Here's a quick graph of the 10 components I'll be selecting for further analysis:
![PCA10](https://github.com/ElishaPhillips/Python-K-Means-RandomForest-Wisconsin-Breast-Cancer-Diagnostics/blob/9c7d169e99817ed944578b93f33bdc127881913a/Graphs/Visualising10.png)

####  K-Means Clustering

I used K-Means clustering to fit the 10 principal components into 2 clusters. The predictive accuracy was 91.04% - not too bad! 
I also labeled the incorrect values according to Malignant or Benign. This is the scatterplot of my K-cluster analysis with incorrect values labeled.

![KMeansAnalysis](https://github.com/ElishaPhillips/Python-K-Means-RandomForest-Wisconsin-Breast-Cancer-Diagnostics/blob/9c7d169e99817ed944578b93f33bdc127881913a/Graphs/BCWD.KCluster.png)

####  Random Forest with 10 Principal Components

Let's try with Random Forest algorithms now, using the 10 selected components. 

Using scikit to split out the training and testing sets:

> X/y test split at default 25%

> X/y train split at default 25%

>n_estimators=100

>max_depth=4

Scatterplot of the Random Forest analysis with incorrect values labeled:

![RandForestAnalysis](https://github.com/ElishaPhillips/Python-K-Means-RandomForest-Wisconsin-Breast-Cancer-Diagnostics/blob/9c7d169e99817ed944578b93f33bdc127881913a/Graphs/BCWD.RandTree.png)

####  Random Forest Analysis with all features

Using all 30 features in this dataset this time, I repeated the split and training processes here to compare. 

> X/y test split at default 25%

> X/y train split at default 25%

>n_estimators=100

>max_depth=4

Normalized the initial X dataset using standardscaler: 98.24%
______________________________________________________________________________________________________________________________________________

## Second Phase: Investigating the Algorithms

The random sampling approach with bootstrap aggregation does help to minimize variation, particularly useful with datasets with high dimensionality, such as this one. Though traditionally random forest algorithms act as a "black box", let's try opening it and see what's going on. Trying to visualize each tree for each decison would be monumental - I saved 100 decision tree images for just one of the 569 predictions. However, theres an easier way. I extracted the individual prediction paths for each feature for further analysis. The algorithm bases it's final predictive factors on the formula:

Prediction = Bias + feature_1_contribution+ …+ feature_n_contribution

_One of the decision trees for the final model:_

![RandForestAnalysis](https://github.com/ElishaPhillips/Python_K_Means_Random_Forest_Breast_Cancer_Diagnostics/blob/b4e2482c448e0eedb9c5cb142e08f6b105f18161/Graphs/tree_modelb.png)

Thankfully sci kit saves the elements for each leaf node, and using treeinvestigator to extract these I formed a database holding the prediction, bias, and left and right contributions per prediction.

Aggregating the contributions per feature across the 569 predictions:

![RandForestAnalysis](https://github.com/ElishaPhillips/Python_K_Means_Random_Forest_Breast_Cancer_Diagnostics/blob/b4e2482c448e0eedb9c5cb142e08f6b105f18161/Graphs/ModelA.png)


![RandForestAnalysis](https://github.com/ElishaPhillips/Python_K_Means_Random_Forest_Breast_Cancer_Diagnostics/blob/b4e2482c448e0eedb9c5cb142e08f6b105f18161/Graphs/ModelB.png)

And here's the contribution by percentage for the final model:

![RandForestAnalysis](https://github.com/ElishaPhillips/Python_K_Means_Random_Forest_Breast_Cancer_Diagnostics/blob/2bd478a5889ff273bd6975ff118979c8387ca8ff/Graphs/ModelBPercentage.png)



