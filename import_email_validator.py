import email_validator

from email_validator import validate_email, EmailNotValidError
import streamlit as st
import pandas as pd

st.title('Email Validation')

st.subheader('By Abdel Hossain')


def check(email):
	try:
	# validate and get info
		v = validate_email(email)
		# replace with normalized form
		email = v["email"]
		print("True")
	except EmailNotValidError as e:
		# email is not valid, exception message is human-readable
		print(str(e))


emailinput = st.text_input('Enter your email here', 'Life of Brian')


st.write('Email Status:', validate_email(emailinput))

