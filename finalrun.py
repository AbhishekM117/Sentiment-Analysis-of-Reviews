# Data handling and processing
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as pyplot
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Statistics
from scipy import stats
import statsmodels.api as sm
from scipy.stats import randint as sp_randint
from time import time

# NLP
import nltk
nltk.download('wordnet')
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


dataset = pd.read_csv("C:/Users/aweso/Desktop/intern/WomensClothingE-CommerceReviews1.csv")
dataset.head()

dataset['Review Text'].fillna('unknown', inplace=True)

reduced_dataset = dataset[['Clothing ID', 'Review Text', 'Recommended IND','Age','Class Name','Rating']]
reduced_dataset.columns = ['CID', 'Review', 'Recommend','age','Class','Rating']



stop = text.ENGLISH_STOP_WORDS

def remove_noise(text):
    
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    
    # Remove special characters
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    
    # Remove punctuation
    text = text.str.replace('[^\w\s]', '')
    
    # Remove numbers
    text = text.str.replace('\d+', '')
    
    # Remove Stopwords
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    # Convert to string
    text = text.astype(str)
        
    return text
reduced_dataset['Filtered Review Text'] = remove_noise(reduced_dataset['Review'])


def sentiment_analyser(text):
    return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))

# Applying function to reviews
reduced_dataset['Polarity'] = sentiment_analyser(reduced_dataset['Filtered Review Text'])

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

reduced_dataset['Filtered Review Text'] = reduced_dataset['Filtered Review Text'].apply(lemmatize_text)


cvec = CountVectorizer(min_df=.005, max_df=.9, ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False)
cvec.fit(reduced_dataset['Filtered Review Text'])

cvec_counts = cvec.transform(reduced_dataset['Filtered Review Text'])

occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Occurrences': occ})
counts_df.sort_values(by='Occurrences', ascending=False)

# Drop all columns not part of the text matrix
ml_model = reduced_dataset

# Create X & y variables for Machine Learning
X = ml_model[['age','Rating']]
y = ml_model['Recommend']

# Create a train-test split of these variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

def cloth_recommendation(age,rating):
    neigh = NearestNeighbors(n_neighbors=100)
    neigh.fit(X_train,y_train) 
    recomended_items=neigh.kneighbors([[age,rating]],return_distance=False)
    recomended_items=recomended_items.transpose()
    recomended_items=recomended_items.flatten()
    nearest_data=reduced_dataset.loc[recomended_items,:]
    high_rated_df=nearest_data[nearest_data['Rating']>=rating]
    final_output=high_rated_df[high_rated_df['Recommend']==1]  
    sorted_output=final_output.sort_values(by=['Polarity'],ascending=False)
    #Displays a Histogram
    sorted_output['age'].hist()
    #Displays a Piechart of Age Distribution
    sorted_output['age'].value_counts().plot(kind='pie',title='Age Distribution',autopct="%1.1f%%")
    pyplot.show()
    

    return sorted_output

from tkinter import *
 
from tkinter.ttk import *
from tkinter import messagebox


window = Tk()

 
window.title("Predictor")
 
window.geometry('1000x1000+300+0')
 
lbl = Label(window, text="Enter your Age:",font = ("Times 16 bold",20))
lbl.grid(column=0, row=0)
txt = Entry(window,width=10)
txt.grid(column=1, row=0)

lbl2 = Label(window, text="Enter the minimum rating:",font = ("Times 16 bold",20))
lbl2.grid(column=0, row=2)
txt2 = Entry(window,width=10)
txt2.grid(column=1, row=2)

#tex=Text(window)
#tex.grid(column=1,row=3)
text1 = Text(window)
text1.grid(column=1,row=3)

def clicked():

    a=int(txt.get())
    b=int(txt2.get())
    recomended_data=cloth_recommendation(a,b)
    
    text1.insert(END, str(recomended_data.iloc[:10,0]))
    
#text1.pack()
          
    

btn = Button(window, text="Submit", command=clicked)
 
btn.grid(column=1, row=8)

window.mainloop()

    
neigh = NearestNeighbors(n_neighbors=100,radius=0.1)
neigh.fit(X_train,y_train) 
recomended_items=neigh.kneighbors([[40,5]],return_distance=False)
recomended_items=recomended_items.transpose()
recomended_items=recomended_items.flatten()
nearest_data=reduced_dataset.loc[recomended_items,:]
high_rated_df=nearest_data[nearest_data['Rating']>=0]
final_output=high_rated_df[high_rated_df['Recommend']==1]  
sorted_output=final_output.sort_values(by=['Polarity'],ascending=False)


fig = pyplot.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

   # Generate the values
x_vals = sorted_output['age']
y_vals = sorted_output['Rating']
z_vals = sorted_output['Polarity']

  # Plot the values
ax.scatter(x_vals, y_vals, z_vals, c = 'b', marker='o')
ax.set_xlabel('age')
ax.set_ylabel('Rating')
ax.set_zlabel('Polarity')

pyplot.show()









































