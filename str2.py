




import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s =pd.read_csv("C:/Users/abdel/Downloads/social.csv", na_values = 'UNKNOWN')

s.isnull().sum()



st.write(s.head(1))
 
#Q2
def clean_sms(inputvar):
    import numpy as np
    x = np.where(inputvar == 1,1,0)
    return(x)
toy_f = pd.DataFrame({'toys':['Barbie','AirsoftGun','Pet Rock'], 'Price':[12.34,1,1]})
clean_sms(toy_f['Price'])



ss_df = pd.DataFrame({
    'sm_li':clean_sms(s['web1h']),
    'income': np.where(s['income']> 9 ,np.nan,s['income'] ),
     'education': np.where(s['educ2'] > 8, np.nan, s['educ2']),
     'parent': clean_sms(s['par']),
     'married': clean_sms(s['marital']),
     'female': np.where( s['gender'] == 2,1,0),
    'age': np.where(s['age'] < 99, s['age'], np.nan)
})

df = ss_df.dropna()


y = df['sm_li']

x  = df.drop(['sm_li'], axis  = 1)

x_train, x_test, y_train,y_test = train_test_split(x,
                                                  y,
                                                 stratify = y,
                                                 test_size = 0.2,
                                                 random_state = 987)


lr = LogisticRegression(class_weight = 'balanced' )

#fitting the data
lr.fit(x_train,y_train)         

LogisticRegression(class_weight='balanced')

y_pred = lr.predict(x_test)


confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))


# Q8
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")




num1 = st.slider(label="Enter an Income", 
          min_value=1,
          max_value=9,
          value=7)

num2 = st.slider(label="Enter an Education",
          min_value=1,
          max_value=10,
          value=1)

num3 = st.slider(label="Parent", 
          min_value=1,
          max_value=8,
          value=5)

num4 = st.slider(label="Female?", 
          min_value=1,
          max_value=8,
          value=5)


num5 = st.slider(label="Married?", 
          min_value=1,
          max_value=8,
          value=5)

num6 = st.slider(label="Age", 
          min_value=1,
          max_value=99,
          value=5)


person = [num1, num2, num3,num4,num5, num6]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

st.write((f"Predicted class: {predicted_class[0]}")) # 0=not pro-environment, 1=pro-envronment
st.write((f"Probability that this person is a linkedin user : {probs[0][1]}")) ## 72.62 percent likeliehood of being a linkedInUser