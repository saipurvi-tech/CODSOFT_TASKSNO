# CODSOFT Data Science Internship Tasks

This repository contains the projects and tasks that i completed during my Data Science Internship at **CodSoft**.

---

### Task 1: Titanic Survival Prediction

**Overview:**  
A model that predicts whether a passenger survived the Titanic shipwreck based on demographic features and ticket parameters.

* **Data Cleaning & EDA:** Handled missing values by imputing `Age` with the column mean and `Embarked` with the mode, while dropping the high-null `Cabin` column. Encoded categorical strings (`Sex` and `Embarked`) into numerical format.
* **Model Training:** Built a binary classification model using Logistic Regression in Scikit-Learn.
* **Evaluation & Interface:** Evaluated accuracy on test data (~78%) and implemented an interactive CLI prompt to predict survival for custom passenger inputs.

---

### Task 2: Movie Recommendation System

**Overview:**  
A content-based recommendation system that suggests similar movies based on textual metadata features using natural language processing techniques and vector similarity.

* **Feature Engineering:** Selected key metadata features (`Name`, `Genre`, `Director`, `Rating`, `Actor 1`, `Actor 2`, `Actor 3`), handled missing values, and combined them into a unified feature string.
* **Vectorization & Similarity:** Transformed textual metadata into numeric vectors using TF-IDF Vectorizer (`TfidfVectorizer`) and computed pairwise similarities using Cosine Similarity.
* **Fuzzy Matching & User Interaction:** Integrated Python's `difflib.get_close_matches` to handle user typos when searching for titles, returning the top 10 most similar movie recommendations in an interactive loop.

---

### Task 3: Iris Flower Classification Model

**Overview:**  
A supervised classification model that predicts the species of an Iris flower (`setosa`, `versicolor`, or `virginica`) using sepal and petal dimensions.

* **EDA & Visualization:** Analyzed dataset distribution, removed duplicate rows, and analyzed feature scatter plots to determine linear separability across species clusters.
* **Model Training:** Split the dataset (80% train / 20% test) and trained a K-Nearest Neighbors (KNN) classifier (K=3) using Scikit-Learn.
* **Performance:** Achieved 100% accuracy on unseen test data. Built a user-input loop that maps custom measurements into a Pandas DataFrame structure for real-time predictions.
