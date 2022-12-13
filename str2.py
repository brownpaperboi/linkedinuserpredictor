
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.title('LinkedIn User Predictor')

st.subheader('By Abdel Hossain')


if  st.button('What is this?') == 1:
    st.write(' Enter in information about yourself and see whether my model can tell if you are a LinkedIn user or not!')


s = pd.read_csv("social.csv", na_values = 'UNKNOWN')

s.isnull().sum()


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

## Income Code 
num1 = 0;
inc = st.number_input('Enter Income', 0, 1000000000)

if inc < 10000:
     num1 = 1
elif inc >= 10000 and inc < 20000:
     num1 = 2
elif inc >= 20000 and inc < 30000:
     num1 = 3
elif inc >= 30000 and inc < 40000:
     num1 = 4
elif inc >= 40000 and inc < 50000:
     num1 = 5
elif inc >= 50000 and inc < 60000:
     num1 = 6
elif inc > 70000 and inc < 80000:
     num1 = 7
elif inc > 80000 and inc < 90000:
     num1 = 8
elif inc >= 90000:
     num1 = 9


# education Code
num2 = 0;
educ = st.selectbox("Choose Highest Education level Achieved", 
              options = ["Grades 1-8 or no formal schooling",
                         "High school incomplete",
                       "High school graduate",
                      "Some college, no degree (includes some community college)",
                      "Associate degree",
                      "Bachelor’s degree",
                      "Some postgraduate, no postgraduate degree (e.g. some graduate school)",
                      "Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)" ])



if educ == "Grades 1-8 or no formal schooling":
     num2 = 1
elif educ == "High school incomplete":
     num2 = 2
elif educ == "High school graduate":
     num2 = 3
elif educ == "Some college, no degree (includes some community college)":
     num2 = 4
elif educ == "Associate degree":
     num2 = 5
elif educ == "Bachelor’s degree":
     num2 = 6
elif educ == "Some postgraduate, no postgraduate degree (e.g. some graduate school)":
     num2 = 7
else:
    num2 = 8




## Parent
num3  = 0

par = st.radio('Are You a Parent?', ['Yes', 'No']);

if par == 'Yes':
    num3 = 1
else:
    num3 = 0 

## Female 

num4 = 0

gen = st.radio('Do you identify as female?', ['Yes', 'No']);

if gen == 'Yes':
    num4 = 1
else:
    num4 = 0 


## Marriage
num5 = 0
mar = st.radio('Are You Married?', ['Yes','No'])

if mar == 'Yes':
    num5 = 1
else:
    num5 = 0 

## Age 
num6 = st.number_input('Enter your age (if greater than 98 please put 98)', 0, 98)

person = [num1, num2, num3,num4,num5, num6]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

if  st.button('Click to see prediction!') == 1:
    if predicted_class[0] > 0:
        st.write((f'You are predicted to be a LinkedIn user, the likelihood that you are using LinkedIn is: {round(round(probs[0][1],3)*100,3)} percent'))
    else:
        st.write((f'You are predicted to be NOT be a LinkedIn user, the likelihood that you are using LinkedIn is {round(round(probs[0][1],3)*100,3)} percent'))
