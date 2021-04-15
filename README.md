
# Decomposing Random Forest Algorithms: Predictive Analysis for Breast Cancer Diagnostics 
### K-Means Clustering and Random Forest Algorithms in Python with JupyterLab

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

####  Initial Random Forest Analysis

There are 30 features in this dataset, I used scikit to split out the training subsets to do an initial random forest analysis as a benchmark for future improvements. 

> X/y test split at default 25%

> X/y train split at default 25%

>n_estimators=100

>max_depth=4

After training and testing my model on the test set the accuracy held at 95%, but unfortunately the model only held 37% accuracy after applying to the full initial dataset. Why did it overfit the data so severely? 

Though Random Forest algorithms are known for overfit resistance due to their implementation of bootstrap aggregation, they can still be affected if a large enough section of the features are highly correlated or if the distribution is imbalanced. They tend to work best with independent features for each decision tree. The features in this dataset, specifically nuclei measurements such as perimeter, area, and radius are split even further into the max, mean, and standard deviation. That most likely accounts for the colinearity that's affecting the algorithm's accuracy. Let's try to improve this.

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

I used K-Means clustering to fit the 10 principal components into 2 clusters. The predictive accuracy was 91.04% - a big improvement! 
I also labeled the incorrect values according to Malignant or Benign. This is the scatterplot of my K-cluster analysis with incorrect values labeled.

![KMeansAnalysis](https://github.com/ElishaPhillips/Python-K-Means-RandomForest-Wisconsin-Breast-Cancer-Diagnostics/blob/9c7d169e99817ed944578b93f33bdc127881913a/Graphs/BCWD.KCluster.png)

####  Random Forest with 10 Principal Components

I then applied the Random Forest algorithm again, this time using the 10 selected components. I repeated the split and training processes on my final model. The predictive accuracy is now improved to 95.08%! 

Scatterplot of the final Random Forest analysis with incorrect values labeled:

![RandForestAnalysis](https://github.com/ElishaPhillips/Python-K-Means-RandomForest-Wisconsin-Breast-Cancer-Diagnostics/blob/9c7d169e99817ed944578b93f33bdc127881913a/Graphs/BCWD.RandTree.png)

####  Results

By reducing the features to 1/3, I managed to improve my classification algorithm to 95% from  the initially dismal 37%. K-Clusters were effective too, reaching 91% classification accuracy.

______________________________________________________________________________________________________________________________________________

## Second Phase: Investigating the Initial Algorithm

Extracting the individual prediction paths for each feature for further analysis
Prediction = Bias + feature_1_contribution+ …+feature_n_contribution

![RandForestAnalysis](https://github.com/ElishaPhillips/Python_K_Means_Random_Forest_Breast_Cancer_Diagnostics/blob/a12459c25c54722f6b0e0da53e19e3d7ea2e5991/Graphs/ModelA.png)
![RandForestAnalysis](https://github.com/ElishaPhillips/Python_K_Means_Random_Forest_Breast_Cancer_Diagnostics/blob/a12459c25c54722f6b0e0da53e19e3d7ea2e5991/Graphs/ModelB.png)




