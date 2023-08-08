# !python3 -m spacy download en_core_web_lg


import pandas as pd
import numpy as np
import spacy
spacy.cli.download("en_core_web_lg")
spacy.prefer_gpu() # Uncomment to use GPUs
nlp = spacy.load('en_core_web_lg')
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
nltk.download('stopwords')

"""# load_form990s

Load the required file for this link -"https://indiana-my.sharepoint.com/personal/fulton_iu_edu/Documents/Project%20990/Data/Data%20for%20Visualizations%20and%20Network%20Analyses%20(TY2019)/Earlier%20Versions/2019_990_All%20Form%20990%20Filers%20(compiled%20from%202020-2021%20IRS%20File%20Extractions)%20w%20Text%20Vars%20and%20NTEE%20codes%20(2,130%20missing)%20(N%20=%20269,217)_2023.2.14.csv"
"""

#The path to the CSV file containing Form 990 data
filename='2019_990_All Form 990 Filers (compiled from 2020-2021 IRS File Extractions) w Text Vars and NTEE codes (2,130 missing) (N = 269,217)_2023.2.14 (1).csv'

#A list of column names to use from the CSV file
cols = ['business_name',
                         'mission_partI',
                         'mission_partIII',
                         'Dsc',
                         'sched_o_FrmAndLnRfrncDsc_grouped',
                         'sched_o_ExplntnTxt_grouped'
                        ]

#A set of stopwords to use when processing text data.
stops = set(stopwords.words('english'))
stops.update(['spirit', 'religion', 'religious'])

#The path to the CSV file containing religious keywords and synonyms
kw_filename ='Religious_Keywords_and_Synonyms.txt'

# Set the desired number of rows to read
num_rows = 20000

# Read the CSV file with the specified number of rows
form990s = pd.read_csv(filename,
                       parse_dates=['return_created'],
                       usecols=cols + ['return_created', 'ein'],
                       nrows=num_rows)



"""# Data Pre-processing

## FORM990 PRE-PROCESSING

**Merge all the textual information present by each EIN (keep the latest one only by looking at return_created). Before Merging, some preprocessing is done like Punctuation removal, stopword removal and lemmatization of words. Finally we get lemmatized keywords per organization.**
"""

#Remove duplicate ein entries by keeping only latest form990 submission
dup_ein = form990s.sort_values('return_created').duplicated('ein', keep='last')
form990s = form990s[~dup_ein]
#replace the index with ein
form990s.index = form990s.ein
form990s.drop(['ein', 'return_created'], axis=1, inplace=True)

#Fill null values and convert all the column data type to string
form990s = form990s.fillna('')
form990s = form990s.astype('string')
print('1.')
def remove_punc_stop(text):
        # This function is utilized only in clean_df_text()
        """
        Removes punctuation and stop words then lemmatizes a string of text.
        :param text: The string of text to be processed.
        :type text: str
        :return: A string containing the processed text.
        :rtype: str
        """
        # Keep only alphanumeric tokens
        text = text.replace('-', ' ')
        text = text.replace('_', ' ')
        text = text.replace(',', ' ')
        text = text.replace('\.', ' ')
        text = re.sub(r'[^A-Za-z0-9\s]*', '', text) #Remove non-alphanumeric chars and spaces
        # Remove stop words
        token_list = [token for token in text.split(' ') if token not in stops]
        text_string = ' '.join(token_list)
        # Lemmatize each word
        doc = nlp(text_string)
        lemmas = [token.lemma_ for token in doc]
        lemma_string = ' '.join(lemmas)
        return lemma_string

def clean_df_text(df):
        # This function is utilized only in load_form990s()
        """
        Cleans and processes the text data in a DataFrame.
        :param df: The DataFrame containing the text data to be cleaned and processed.
        :type df: pd.DataFrame
        :return: A Series containing the cleaned and processed text data.
        :rtype: pd.Series
        """
        # Select only text columns
        txtcols = cols
        # Merge all the text fields into one column
        df = df[txtcols].agg(' '.join, axis=1)

        # Make all text lower case
        df = df.str.lower()
        # keep only alphanumeric tokens, remove stop words and lemmatize
        df = df.apply(remove_punc_stop)
        return df

form990s = clean_df_text(form990s)

#Save the final form
with open('form990txt1.pkl', 'wb') as f:
    pickle.dump(form990s, f)

"""##  Religious keywords and synonym Pre-processing

**Make a dictionary for English and non-english keywords. Also store lemmatized English keywords**
"""

#Loads religious keywords from a file
keywords = pd.read_csv(kw_filename, encoding_errors = 'replace', header=None).iloc[:,0]

# Remove leading and trailing spaces
keywords = keywords.str.replace("^\s", "")
keywords = keywords.str.replace("\s$", "")
# Remove non-alphanumeric or space characters
keywords = keywords.str.replace("[^A-Za-z0-9\s\']*", "")
# Remove extra spaces around apostraphe
keywords = keywords.str.replace("\s\'\s", "\'")
# Remove double spaces
keywords = keywords.str.replace("\s{2}", " ")
keywords = keywords.tolist()

# Function to identify english vs non-english keywords
def classify_english_terms(keyword):
    """
    Classifies a keyword using a spaCy model.
    :param keyword: The keyword to be classified.
    :type keyword: str
    :return: A boolean indicating whether the spaCy model's vector for the keyword is non-zero.
    :rtype: bool
    """
    spacy_doc = nlp(keyword)
    return spacy_doc.vector.any()

kw_is_english_list = []
# Classify keywords as English vs Non-English
for keyword in keywords:
    kw_is_english = classify_english_terms(keyword)
    kw_is_english_list.append(bool(kw_is_english))

# Store English and Non-English keywords separately
english_kws = pd.Series(keywords)[pd.Series(kw_is_english_list)].tolist()
nonenglish_kws = pd.Series(keywords)[~pd.Series(kw_is_english_list)].tolist()

# lemmatize only english_kws because english model won't understand non_english words
english_lemmas = []
for kw in english_kws:
    doc = nlp(kw)
    temp_lemmas = []
    for token in doc:
        temp_lemmas.append(token.lemma_)
    lemmas = ' '.join(temp_lemmas)
    english_lemmas.append(lemmas)

# Store english and non-english words in dictionary
kw_dict = {}
kw_dict['english_lemmas'] = english_lemmas
kw_dict['nonenglish'] = nonenglish_kws
with open('kw_dict.pkl', 'wb') as f:
    pickle.dump(kw_dict, f)

with open('kw_dict.pkl', 'rb') as f:
    kw_dict = pickle.load(f)

"""# Vectorization of keywords

By setting ngram_range=(1,3), the CountVectorizer will consider unigrams (single words), bigrams (pairs of consecutive words), and trigrams (triplets of consecutive words) during the vectorization process.
"""

### VECTORIZE NONENGLISH KEYWORDS
nonenglish_kws = kw_dict['nonenglish']
cv = CountVectorizer(ngram_range=(1,3))
cv.fit(np.array(nonenglish_kws))
print('2.')
### VECTORIZE ENGLISH KEYWORDS

#-------------------------------------Around 40 minutes ------------------------
def transform_english(X):
    """
    Vectorize English text data using semantic similarity.
    :param X: The Series containing the English text data to be transformed.
    :type X: pd.Series
    :return: A DataFrame containing the transformed text data.
    :rtype: pd.DataFrame
    """
    eng_kws = kw_dict['english_lemmas']
    num_kws = len(eng_kws)

    #eng_vecs will represent how each keyword is respresnted in terms of a vector
    eng_vecs = np.zeros((num_kws, 300))
    for i, kw in enumerate(eng_kws):
        eng_vecs[i] = nlp(kw).vector
        


    def calc_sim_score(text, num_scores=5):
        doc = nlp(text)
        num_tokens = doc.__len__()
        doc_vecs = np.zeros((num_tokens, 300))
        for i, token in enumerate(doc):
            doc_vector = token.vector
            doc_vecs[i] = doc_vector
        sim_scores = cosine_similarity(doc_vecs, eng_vecs)
        # keep num_scores of similarity scores
        sim_score_sort_order = sim_scores.argsort(0)[-num_scores:][::-1]
        sim_scores = np.take_along_axis(sim_scores, sim_score_sort_order, axis=0)
        # Turn sim_scores into 1 record with sim scores of each word adjacent to each other
        sim_scores = sim_scores.T.reshape(1,-1)
        return sim_scores

    X = X.apply(calc_sim_score)

    def reshape(X):
        X = np.reshape(X, newshape=(-1))
        return X

    X = X.apply(reshape)
    idx = X.index
    X = X.values.tolist()
    X = pd.DataFrame(X, index=idx)
    return X

func_transform = FunctionTransformer(transform_english)

transformers = [('english_sim', func_transform),
                        ('nonenglish_vec', cv)
                       ]
feat_union = FeatureUnion(transformers, n_jobs=1)

vec_df=feat_union.transform(form990s)

#Save the final form
with open('form990_vecs1.pkl', 'wb') as f:
    pickle.dump(vec_df, f)
#-------------------------------------Around 2 hours ------------------------   
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Assuming you have a DataFrame called 'vec_df' containing the feature vectors

# Initialize an empty DataFrame to store the silhouette scores
silhouette_scores_df = pd.DataFrame(columns=['NumClusters', 'SilhouetteScore'])

# Define the range of cluster numbers
cluster_range = range(2, 16)

# Iterate over the cluster numbers and compute the silhouette scores
for num_clusters in cluster_range:
    # Create the pipeline with StandardScaler and KMeans clustering
    pipeline = Pipeline([
        ('std', StandardScaler(with_mean=False)),
        ('kmeans', KMeans(n_clusters=num_clusters))
    ])
    
    # Fit the pipeline on the data
    pipeline.fit(vec_df)
    
    # Compute the silhouette score
    silhouette_avg = silhouette_score(vec_df, pipeline['kmeans'].labels_)
    
    # Append the silhouette score to the DataFrame
    silhouette_scores_df = silhouette_scores_df.append({'NumClusters': num_clusters, 
                                                        'SilhouetteScore': silhouette_avg},
                                                       ignore_index=True)

# Save the DataFrame with silhouette scores to a file
silhouette_scores_df.to_csv('silhouette_scores.csv', index=False)

steps = [('std', StandardScaler()),
         ('kmeans', KMeans(n_clusters=24)) # n_clusters
        ]
pipe = Pipeline(steps)
pipe.fit(vec_df.toarray())

# Predict Clusters for Form 990s
cluster_preds = pipe.predict(vec_df.toarray())

# Add Clusters to form 990s
form990s = pd.DataFrame(form990s)
form990s['cluster'] = cluster_preds



orig_filepath = '2019_990_All Form 990 Filers (compiled from 2020-2021 IRS File Extractions) w Text Vars and NTEE codes (2,130 missing) (N = 269,217)_2023.2.14 (1).csv'
orig_form990s = pd.read_csv(
    orig_filepath,
    index_col = 'ein',
    parse_dates=['return_created'],
    nrows=20000
)

# Remove duplicate form 990s by keeping latest submission
orig_form990s.sort_values('return_created', inplace=True)
orig_form990s['ein'] = orig_form990s.index
orig_form990s.drop_duplicates(
    subset='ein',
    keep='last',
    inplace=True
)
orig_form990s.drop('ein', axis=1, inplace=True)

# Select only columns I wish to keep
keep_cols = [
    'business_name',
    'ntee_full'
]
orig_form990s = orig_form990s[keep_cols]

# Add NTEE code and Description columns
orig_form990s['ntee_letter'] = orig_form990s.ntee_full.str[0]
ntee_map = {'A': 'Arts, Culture, and Humanities',
            'B': 'Education',
            'C': 'Environment and Animals',
            'D': 'Environment and Animals',
            'E': 'Health',
            'F': 'Health',
            'G': 'Health',
            'H': 'Health',
            'I': 'Human Services',
            'J': 'Human Services',
            'K': 'Human Services',
            'L': 'Human Services',
            'M': 'Human Services',
            'N': 'Human Services',
            'O': 'Human Services',
            'P': 'Human Services',
            'Q': 'International, Foreign Affairs',
            'R': 'Public, Societal Benefit',
            'S': 'Public, Societal Benefit',
            'T': 'Public, Societal Benefit',
            'U': 'Public, Societal Benefit',
            'V': 'Public, Societal Benefit',
            'W': 'Public, Societal Benefit',
            'X': 'Religion Related',
            'Y': 'Mutual/Membership Benefit',
            'Z': 'Unknown, Unclassified'
           }
orig_form990s['ntee_desc'] = orig_form990s.ntee_letter.map(ntee_map)



# Merge Orig form 990s with cluster labeled form 990s
full_form990s = pd.merge(
    left = form990s,
    right = orig_form990s,
    left_index = True,
    right_index = True,
    how = 'left'
 )



# # Analyze Distribution of NTEE in each cluster
# cluster_counts = full_form990s[['cluster', 'ntee_desc', 'business_name']].groupby(['cluster', 'ntee_desc']).count()
# cluster_counts = cluster_counts.reset_index().pivot('cluster', 'ntee_desc', 'business_name')
# #Percentage based on each cluster
# cluster_props = cluster_counts.div(cluster_counts.sum(1), 0)

# # Save the results to a file
# cluster_props.to_csv('cluster_counts24.csv')




# """Based on the proportion table we can judge which cluster is more religious as I used only 1000 ROWS,  so the ratio is less but it might look effective while using whole dataset"""

# ##NOTE: THIS MAPPING IS dynamic
# NOTE: THIS MAPPING IS dynamic
cluster_map = {
    0: False,
    1: False,
    2: False,
    3: True,
    4: False,
    5: False,
    6: False,
    7: True,
    8: False,
    9: False,
    10: True,
    11: False,
    12: False,
    13: False,
    14: False,
    15: False,
    16: False,
    17: False,
    18: False,
    19: False,
    20: False,
    21: False,
    22: False,
    23: False
}

cluster_TF = full_form990s['cluster'].map(cluster_map)

religious_TF = full_form990s['ntee_letter'].apply(lambda x: True if x == 'X' else False)
full_form990s['is_religious'] = cluster_TF | religious_TF

full_form990s.to_pickle('df1.pkl')
full_form990s.to_csv('full_form990s.csv', index=False)

