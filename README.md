#Music Recommender System using KNN
Overview
This project implements a music recommender system using K-Nearest Neighbors (KNN) to analyze and categorize songs based on various audio features. The dataset used contains top hits from Spotify between 2000 and 2019, providing a rich source of data for music analysis and recommendations.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Fuzzywuzzy
- LIME
- Dataset
The dataset used in this project is sourced from Kaggle's Top Hits from Spotify 2000-2019. It includes features such as:

- Song title
- Artist
- Year
- Genre
- Audio features (e.g., danceability, energy, loudness, etc.)
- Explicit content flag
## Steps Involved
### 1. Data Loading and Initial Exploration

```
music_data = pd.read_csv('/kaggle/input/top-hits-from-spotify-2000-2019/Top Hits from Spotify 2000  2019.csv')
music_data.head()
music_data.isnull().sum()
music_data.info()
```
### 2. Data Cleaning and Preprocessing
Handling Missing Values: Visualized the count of null values.
Converting Explicit Column: Transformed the 'explicit' boolean column to integers.

```
music_data["explicit"] = music_data["explicit"].astype(int)
```
Correlation Analysis: Generated a correlation matrix to understand relationships between features.
### 3. Data Visualization
- Trend Analysis: Visualized trends in music over the years using bar plots for various features.

- Heatmap: Created a heatmap to visualize correlations among numerical features.

### 4. Encoding Categorical Data
Used OneHotEncoder to encode the genre column, enabling the model to handle categorical data effectively.


```
from sklearn.preprocessing import OneHotEncoder
# Encoding genres
```
### 5. Normalization
Normalized the numerical columns to scale the features for better performance in clustering and classification.


```
def normalize_column(col):
    max_d = music_data[col].max()
    min_d = music_data[col].min()
    music_data[col] = (music_data[col] - min_d)/(max_d - min_d)
```
### 6. Outlier Detection
Utilized boxplots to identify and visualize potential outliers in the dataset.

### 7. Dimensionality Reduction with t-SNE
Applied t-SNE to reduce dimensionality and visualize the data in two dimensions.


```
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(music_data_modified)
```
### 8. Clustering
Employed KMeans clustering to categorize songs into clusters based on their audio features.


```
km = KMeans(n_clusters=10)
cat = km.fit_predict(num)
music_data['cat'] = cat
```
### 9. Model Training and Validation
Splitting Data: Divided the dataset into training, validation, and test sets.
KNN Model Training: Trained KNN classifiers with varying values of K.

```
knn5 = KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=5)
```
### 10. Evaluation
Evaluated model performance using accuracy scores and confusion matrices.


```
from sklearn.metrics import accuracy_score
print("Accuracy with k=5", accuracy_score(y_valid, y_pred_5)*100)
```
### 11. Recommendations
Created a recommendation function to suggest songs based on user input, utilizing the KNN model to find similar tracks.


```
def recommender(song_name, data, model):
    # Logic to find and recommend similar songs
```
### 12. Model Explainability with LIME
Used LIME (Local Interpretable Model-agnostic Explanations) to provide insights into model predictions.


```
exp = explainer.explain_instance(X_valid.iloc[0,:].values, knn5.predict_proba, num_features=10)
```
## Visualizations
Generated various plots to visualize trends, correlations, clustering results, and model predictions.
Conclusion
This project effectively demonstrates how to build a music recommender system using KNN, providing insights into data preprocessing, clustering, and model evaluation. Further enhancements could include implementing more advanced recommendation algorithms or expanding the dataset for richer analysis.

## Usage
To run the code, ensure all dependencies are installed and the dataset path is correctly set. Follow the steps outlined in the code sections above to execute each part of the analysis.
