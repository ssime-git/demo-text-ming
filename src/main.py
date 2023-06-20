import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Étape de prétraitement des données
def preprocess(text):
    # Supprimer la ponctuation et les chiffres
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])

    # Mettre en minuscules
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Supprimer les stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Rejoindre les tokens en une seule chaîne de texte
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text

# Étape de représentation des données
def represent(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names()

    return X, feature_names

# Textes d'exemple
texts = [
    "Text mining is the process of extracting useful information from unstructured text data.",
    "The preprocessing step involves cleaning and preparing the text for analysis.",
    "Representation of the data is important to make it suitable for machine learning algorithms.",
]

# Prétraitement des textes
preprocessed_texts = [preprocess(text) for text in texts]

# Représentation des données
X, feature_names = represent(preprocessed_texts)

# Affichage des résultats
print("Textes d'origine :")
for text in texts:
    print("- ", text)
print("\nTextes prétraités :")
for text in preprocessed_texts:
    print("- ", text)
print("\nMatrice de termes :")
print(X.toarray())
print("\nListe des termes :")
print(feature_names)