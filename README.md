# Music Recommender System using KNN
## Overview
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

## Dataset

The dataset used in this project is sourced from Kaggle's Top Hits from Spotify 2000-2019. It includes features such as:

- Song title
- Artist
- Year
- Genre
- Audio features (e.g., danceability, energy, loudness, etc.)
- Explicit content flag

## Workflow
### Data Cleaning and Preprocessing
```python
music_data["explicit"] = music_data["explicit"].astype(int)
```
### Normalization
```python
def normalize_column(col):
    max_d = music_data[col].max()
    min_d = music_data[col].min()
    music_data[col] = (music_data[col] - min_d)/(max_d - min_d)
```
### K-Means Clustering
```python
km = KMeans(n_clusters=10)
cat = km.fit_predict(num)
music_data['cat'] = cat
```
### Training and Validation
```python
knn5 = KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=5)
```
### Evaluation
```python
from sklearn.metrics import accuracy_score
print("Accuracy with k=5", accuracy_score(y_valid, y_pred_5)*100)
```

## Conclusion
Works
