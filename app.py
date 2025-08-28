import streamlit as st
import pandas as pd 
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#login page 
st.set_page_config(page_title='Depression Detection',page_icon=':guardsman:',layout='centered')
def login_page():
    st.title('Mood Sense')
    st.write('MoodSense is an AI-powered application that helps detect early signs of depression through text analysis. It provides users with meaningful insights into their emotional well-being, encouraging timely support and self-care.')
    st.write('Enter credentials to continue.')
    
    st.markdown("""
<div style="background-color:#2d2d2d; padding:15px; border-radius:10px; border:1px solid #444;">
    <h4 style="color:#f1c40f;">🔑 Temporary Login Credentials</h4>
    <p><b style="color:#ecf0f1;">Username:</b> <code style="color:#2ecc71;">username</code></p>
    <p><b style="color:#ecf0f1;">Password:</b> <code style="color:#2ecc71;">password</code></p>
</div>
""", unsafe_allow_html=True)
    
    username=st.text_input('Enter username')
    password=st.text_input('Enter password',type='password')
    if st.button('Login'):
        if username=='username' and password=='password':
            st.session_state['logged_in']=True #this tells that abhi new login hua hai and agar user dubara aaega toh log out nahi hoga
            st.success('Login successful')
            return main_page()
        else:
            st.error('Login failed')



def main_page():
    #app page
    st.title('Depression Detection')


    #taking the input from user
    text=st.text_input(label='Enter the text to analyse depression type',max_chars=200,placeholder='Enter text')

    #loading the tokenizer
    tfidf=pickle.load(open('tfidf.pickle','rb'))
    #loading the model
    model=pickle.load(open('model.pickle','rb'))

    if text is not None and text!='':
        #making prediction
        vectorized_text=tfidf.transform([text])
        result=model.predict(vectorized_text)[0]

        #showing the predicted label
        st.write(f'The predicted depression type is : {result}')

        #showing the prediction probability
        st.write(f'The prediction probability is : {model.predict_proba(vectorized_text).max().round(2)}')
    else:
        st.warning('Please enter the text to analyse')
        st.stop()



    #showing the prediction for each class
    results=[]
    for i in model.classes_:
        results.append(model.predict_proba(vectorized_text)[0][list(model.classes_).index(i)])

    df=pd.DataFrame({'Depression Type':model.classes_,'Probability':results})
    st.dataframe(df)

    #making a bar chart
    st.bar_chart(data=df,x='Depression Type',y='Probability')

# ---- App Flow ----
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    main_page()
else:
    login_page()