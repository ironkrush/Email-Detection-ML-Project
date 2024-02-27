import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st


data = pd.read_csv('spam.csv')
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

mess = data['Message']
cat = data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size=0.2)


cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)


#model

model = MultinomialNB() 
model.fit(features, cat_train)

#testing our model
features_test = cv.transform(mess_test)

#checking prediction power 
def predict(message):
            input_message = cv.transform([message]).toarray()
            result = model.predict(input_message)
            return result

#coding for web
st.header('Spam Detection')

output = predict('Congratulations, you won a lottery')
input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
      output =  predict(input_mess)
      st.markdown(output)