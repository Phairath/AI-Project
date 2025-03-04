import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
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
st.sidebar.success('**Explore** a [Titanic Dataset](%s)' %'https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
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
st.write('### Supervised Learning use Ensemble Learning Method')
st.markdown('#### K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree')
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
    df_titanic = load_data('https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv')
    # st.dataframe(df_titanic)
    num_shown = st.slider('Slide to show data: ',min_value=5,max_value=len(df_titanic),value=5)
    st.dataframe(df_titanic[:num_shown])
    fig1 = px.box(df_titanic,x=['Age'],
                hover_data=['Survived'],title='Data',width=700)
    fig2 = px.box(df_titanic,x=['SibSp'],
                hover_data=['Survived'],title='Data',width=700)
    st.write(fig1)
    st.write(fig2)
    q1 = df_titanic['Age'].quantile(0.25)
    q3 = df_titanic['Age'].quantile(0.75)
    iqr = q3-q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_titanic[(df_titanic['Age'] < lower_bound) | (df_titanic['Age'] > upper_bound)]
    st.write(outliers)
    st.write('### Number of Survived: ',df_titanic[df_titanic['Survived'] == 1].shape[0],'/',df_titanic.shape[0])


if page_selected == 'Model':
    cols = st.columns(2)
    with cols[0]:
        st.write('Input your data for predict')
        pClass = st.number_input('Passenger Class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)',min_value=1,max_value=3)
        sex = st.number_input('Sex (1 = Male, 2 = Female)',min_value=1,max_value=2)
        age = st.number_input('Age',min_value=1)
        sibSp = st.number_input('Siblings/Spouses Aboard',min_value=0)
        parch = st.number_input('Parents/Children Aboard',min_value=0)
        btn1 = st.button('Predict!')
        if btn1:
            st.write('Prediction not availible now. Pls wait!!!')
        
    with cols[1]:
        st.write('### Data Description (ไม่จำเป็นต้องใส่)')
        df_description = pd.DataFrame({
            'Variable': ['Passenger Class','Age'],
            'Definition': ['Ticket class','Age in years']
        })
        st.table(df_description)

        code = '''print('hello')'''
        st.code(code,language='python')
        btn1 = st.button('Run!')
        if (btn1):
            st.write('haha')

    



        
    