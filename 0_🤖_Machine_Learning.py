import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
# import pages.page1 as page1
st.set_page_config(
    page_title = 'Machine Learning',
    page_icon = '🤖',
    initial_sidebar_state = 'auto',
    layout = 'centered' 
)

with st.sidebar:
    page_selected = option_menu('Pages', ['Main', 'Model'], 
        menu_icon="list-ul", icons = ['house', 'box'], default_index=0,orientation = 'horizontal')
    # page_selected

# page_option = st.sidebar.radio('Choose Page', ['Main','Model'])

# st.sidebar.success("Machine Learning")
st.sidebar.success('**Explore** the [Titanic Dataset](%s)' %'https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
# st.sidebar.markdown("<br>"*6, unsafe_allow_html=True)


st.sidebar.success('**Created by** Phairath Jaradnaparatana 6604062630382 SEC.2')
st.sidebar.success("**Powered by** [Streamlit](https://streamlit.io/)")

# st.sidebar.markdown("""
#     <p style="text-align: center; font-size: 16px; color: gray;">Created by Phairath Jaradnaparatana 6604062630382 SEC.2</p>
#     """, unsafe_allow_html=True)
# st.sidebar.markdown("""
#     <p style="text-align: center; font-size: 16px; color: gray;">Powered by Streamlit</p>
#     """, unsafe_allow_html=True)
# st.sidebar.markdown("**Powered by Streamlit**")


st.title('🚢 Titanic Survival Simulator: Test Your Fate with Data!')
st.write('#### \# Supervised learning leverages Ensemble Learning techniques to enhance model performance')
st.markdown('#### \# K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree')
st.markdown('<span style="color: red; font-size: 20px;">\****Information about the dataset and other details can be found in the sidebar**</span> ', unsafe_allow_html=True)
# st.markdown(
#     """
#     <style>
#     .title {
#         text-align: center;
#         font-size: 40px;
#         font-weight: bold;
#     }
#     </style>
#     <div class="title">
#         My Streamlit App Title
#     </div>
#     """, 
#     unsafe_allow_html=True
# )

@st.cache_data(persist=True)
def load_data(url):
    # df_titanic = pd.read_csv('./data/titanic.csv')
    df = pd.read_csv(url)
    return df

if page_selected == 'Main':
    st.write('## Step 1: Exploratory Data Analysis (EDA)')
    df_titanic = load_data('https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv')
    # st.dataframe(df_titanic)
    num_shown = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_titanic),value=5,key='slider1')
    st.dataframe(df_titanic[:num_shown])
    st.write('### - Data Description (Focus on Abbreviated Features)')
    df_description = pd.DataFrame({
        'Variable': ['Pclass','Parch','SibSp','Embarked'],
        'Definition': ['Ticket class','\# of parents / children aboard the Titanic',
                        '\# of siblings / spouses aboard the Titanic',
                        'Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)']
    })
    st.table(df_description)
    st.write('### - Remove irrelevant columns for model training')
    with st.echo():
        df_titanic2 = df_titanic.drop(columns=['PassengerId','Name','Ticket','Cabin'])
    st.write('### - Remove Missing Values')
    with st.echo():
        df_clean_titanic = df_titanic2.dropna()
    num_shown2 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_clean_titanic),value=5,key='slider2')
    st.dataframe(df_clean_titanic[:num_shown2])
    st.write('### - Transform data')
    st.markdown('**Data Encoding**')
    with st.echo():
        df_clean_titanic['Sex'] = df_clean_titanic['Sex'].map({'male': 0,'female': 1})
        df_clean_titanic['Embarked'] = df_clean_titanic['Embarked'].map({'C': 0,'Q': 1,'S': 2})
    st.markdown('**Data Scaling**')
    with st.echo():
        scaler = MinMaxScaler()
        df_clean_titanic['Pclass'] = scaler.fit_transform(df_clean_titanic[['Pclass']])
        df_clean_titanic['Age'] = scaler.fit_transform(df_clean_titanic[['Age']])
        df_clean_titanic['SibSp'] = scaler.fit_transform(df_clean_titanic[['SibSp']])
        df_clean_titanic['Parch'] = scaler.fit_transform(df_clean_titanic[['Parch']])
        df_clean_titanic['Fare'] = scaler.fit_transform(df_clean_titanic[['Fare']])
        df_clean_titanic['Embarked'] = scaler.fit_transform(df_clean_titanic[['Embarked']])
    num_shown3 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_clean_titanic),value=5,key='slider3')
    st.dataframe(df_clean_titanic[:num_shown3])
    st.write('### - Visualize and Remove Outliers')
    fig1 = px.box(df_clean_titanic,x=['Age'],
                hover_data=['Survived'],title='Age',width=700)
    fig2 = px.box(df_clean_titanic,x=['SibSp'],
                hover_data=['Survived'],title='SibSp',width=700)
    fig3 = px.box(df_clean_titanic,x=['Parch'],
                hover_data=['Survived'],title='Parch',width=700)
    fig4 = px.box(df_clean_titanic,x=['Fare'],
                hover_data=['Survived'],title='Fare',width=700)
    st.write(fig1)
    with st.echo():
        df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['Age'] >= 0.824].index, inplace=True)
    st.write(fig2)
    with st.echo():
        df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['SibSp'] >= 0.6].index, inplace=True)
    st.write(fig3)
    with st.echo():
        df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['Parch'] >= 0.5].index, inplace=True)
    st.write(fig4)
    with st.echo():
        df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['Fare'] >= 0.138].index, inplace=True)
    # q1 = df_clean_titanic['Age'].quantile(0.25)
    # q3 = df_clean_titanic['Age'].quantile(0.75)
    # iqr = q3-q1
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr
    # outliers = df_clean_titanic[(df_clean_titanic['Age'] < lower_bound) | (df_clean_titanic['Age'] > upper_bound)]
    # st.write(outliers)
    num_shown4 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_clean_titanic),value=5,key='slider4')
    st.dataframe(df_clean_titanic[:num_shown4])
    st.write('### Number of Survived: (ควรใส่ไหม ??)',df_clean_titanic[df_clean_titanic['Survived'] == 1].shape[0],'/',df_clean_titanic.shape[0])
    st.write('---')
    st.write('## Step 2: Model Training')


    

if page_selected == 'Model':
    cols = st.columns(2)
    with cols[0]:
        st.write('Input your data for predict')
        pClass = st.selectbox('Passenger Class',('1st class','2nd class','3rd class'))
        # st.number_input('Passenger Class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)',min_value=1,max_value=3)
        sex = st.selectbox('Sex',('Male','Female'))
        # st.number_input('Sex (1 = Male, 2 = Female)',min_value=1,max_value=2)
        age = st.number_input('Age',min_value=1)
        sibSp = st.number_input('Siblings/Spouses Aboard',min_value=0)
        parch = st.number_input('Parents/Children Aboard',min_value=0)
        btn1 = st.button('Predict!')
        if btn1:
            st.write('Prediction not availible now. Pls wait!!!')
        
    with cols[1]:
        st.write('### Data Description')
        df_description = pd.DataFrame({
            'Variable': ['Pclass','Age','Parch','SibSp','Embarked'],
            'Definition': ['Ticket class','Age in years','\# of parents / children aboard the Titanic',
                           '\# of siblings / spouses aboard the Titanic',
                           'Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)']
        })
        st.table(df_description)

        code = '''print('hello')'''
        st.code(code,language='python')
        btn1 = st.button('Run!')
        if (btn1):
            st.write('haha')

    



        
    