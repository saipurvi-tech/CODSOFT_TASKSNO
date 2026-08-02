import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def EDA_and_ModelTraining():
    #In this part, we are analysing our data by performing EDA.
    df = pd.read_csv('IRIS.csv')
    #print(df.head()) 
    #we get to know that there are 5 columns in the dataset, namely: sepal_length, sepal_width, petal_length, petal_width and species.

   # print(df.shape)  
    #we get to know that there are 150 rows and 5 columns in the dataset. 

    #print(df.info())
    #in this, we find out the RangeIndex, Data Columns, the Non-Null Count, Their DType (which is float)  and the memory useage (which is 6.0KB)

    #print(df.describe())
    #Here, we were able to see the count, mean, std, min, percentile values, max values of sepal_length, sepal_width, petal_width and petal_length. 

    #print(df.isnull().sum())
    #We get to know that there are no null values peresent in our dataset. 

    data = df.drop_duplicates(subset = 'species')

    #print(df.value_counts('species')) 
    #we get to know that there are only 3 species - senota, versicolour, and virginica. Each species has 50 rows in the dataset. 
    #print('\n\n')


    #From here, we are training our model. we are using KNN classifer to train our model.
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] 
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(X_train, y_train)

    #y_pred = model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

    return model, X.columns #returnign our trained model.


def main():
    model, columns = EDA_and_ModelTraining()
    sample_data = input("\nEnter the measurements (sepal_length, sepal_width, petal_length, petal_width) : ")
    sample_data = [float(x) for x in sample_data.split(',')]
    sample_df = pd.DataFrame([sample_data], columns=columns)
    predicted_species = model.predict(sample_df)

    print(f'\nSample Input: {sample_data}')
    print(f'\nPredicted Species: {predicted_species[0]}')

if __name__ == "__main__":
    while True:
        main()
        print("\n-----------------------------------------------------------------------------")
        continue_choice = input("\n\nDo you want to continue? (yes/no): ")
        if continue_choice.lower() != 'yes':
            break