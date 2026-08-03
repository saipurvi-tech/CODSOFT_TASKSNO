import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#loading the dataset from the CSV file to a pandas dataframe
movies_data = pd.read_csv('movies.csv', encoding='latin1')

#print(movies_data.head())

#we are removing the unwanted special characters from the beginning of movie names
movies_data['Name'] = movies_data['Name'].str.replace(r'^[^A-Za-z0-9]+', '', regex=True)

#print(np.shape(movies_data)) #we find out that there are 15509 rows and 10 columns in our dataset. 

#now we are going to select only the revelant columns for our recomendation system. 
#print(movies_data.columns) 
#The columns present are:  Index(['Name', 'Year', 'Duration', 'Genre', 'Rating', 'Votes', 'Director''Actor 1', 'Actor 2', 'Actor 3'], dtype='str')
selected_features = ['Name', 'Rating', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
#print(selected_features) 
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('').astype(str) #replacing null value with null stringin the selected features.

#now we are combinng all the selected features
combined_features = movies_data['Name']+' '+movies_data['Rating']+' '+movies_data['Genre']+' '+movies_data['Director']+' '+movies_data['Actor 1']+' '+movies_data['Actor 2']+' '+movies_data['Actor 3']
#print(combined_features.head())

#we are going to convert the text data into feature vector using TfidfVectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
#print(feature_vectors)

#now we are going to find the similarity score using cosine similarity
similarity = cosine_similarity(feature_vectors)
#print(similarity)
#print(similarity.shape)

while True:
    #getting the movie name from the user
    movie_name = input("\n\nEnter your favourite movie name: ")

    #creating a list with all the movie names given in the dataset
    list_of_all_titles = movies_data['Name'].tolist()
    #print(list_of_all_titles)

    #finding the closest match for the movie name given by the user
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    #print(find_close_match)
    close_match = find_close_match[0]
    #print(close_match)

    #finding the index along with the movie name
    index_of_the_movie = [movies_data[movies_data['Name'] == close_match].index[0]]
    #print(index_of_the_movie)

    #getting a list of similar movies, based on the similarity score
    similarity_score = list(enumerate(similarity[index_of_the_movie[0]]))
    #print(similarity_score)
    #print(len(similarity_score))

    #now we are going to sort the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x:x[1], reverse = True)
    #print(sorted_similar_movies)

    #printing the names of similar movies based on the index
    print("Movies that we suggested for you: \n")
    i=1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['Name'].values[0]
        if (i<11):
            print(i, '.', title_from_index)
            i += 1

    user_input = input("\nDo you want to continue? (yes/no): ")
    if user_input.lower() != 'yes':
        break