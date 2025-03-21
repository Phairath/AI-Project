import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(
    page_title="Neural Network",
    page_icon="🧠",
    initial_sidebar_state = 'auto',
    layout = 'centered'
)

with st.sidebar:
    page_selected = option_menu('Pages', ['Main', 'Model'], 
        menu_icon="list-ul", icons = ['house', 'box'], default_index=0,orientation = 'horizontal')

st.sidebar.success('**Explore** the [Road Accident Dataset](%s)' %'https://www.kaggle.com/datasets/himelsarder/road-accident-survival-dataset')
st.sidebar.success('**Created by** [Phairath Jaradnaparatana 6604062630382 SEC.2](%s)' %'https://github.com/Phairath')
st.sidebar.success("**Powered by** [Streamlit](https://streamlit.io/)")

st.title('🚗 Road Accident Survival Simulator: Test Your Fate with Data!')
st.markdown('<span style="color: red; font-size: 20px;">\****Information about the dataset and other details can be found in the sidebar**</span> ', unsafe_allow_html=True)

@st.cache_data(persist=True)
def load_data():
    try:
        df = pd.read_csv('./data/accident.csv')
    except Exception:
        st.write('Error to Open the Dataset')
    return df

if page_selected == 'Main':
    st.write('## Step 1: Exploratory Data Analysis (EDA)')
    df_accident = load_data()
    num_shown = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_accident),value=5,key='slider1')
    st.dataframe(df_accident[:num_shown])

    st.write('### - Data Description')
    df_description = pd.DataFrame({
        'Variable': ['Age','Gender','Speed_of_Impact','Helmet_Used','Seatbelt_Used','Survived'],
        'Data Type' : ['Continuous','Nominal','Continuous','Nominal','Nominal','Nominal'],
        'Definition': ['Age in years','Gender',
                        'Velocity or Speed at which two objects collide',
                        'Helmet use','Seatbelt use','Survival']
    })
    st.table(df_description)

    st.write('### - Transform data')
    st.markdown('**Data Encoding**')
    with st.echo():
        df_accident['Gender'] = df_accident['Gender'].map({'Male': 0,'Female': 1})
        df_accident['Helmet_Used'] = df_accident['Helmet_Used'].map({'No': 0,'Yes': 1})
        df_accident['Seatbelt_Used'] = df_accident['Seatbelt_Used'].map({'No': 0,'Yes': 1})

    st.markdown('**Data Scaling**')
    with st.echo():
        scaler = MinMaxScaler()
        df_accident['Age'] = scaler.fit_transform(df_accident[['Age']])
        df_accident['Speed_of_Impact'] = scaler.fit_transform(df_accident[['Speed_of_Impact']])
    num_shown2 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_accident),value=5,key='slider2')
    st.dataframe(df_accident[:num_shown2])

    st.write('### - Remove Missing Values')
    with st.echo():
        df_accident = df_accident.dropna()
    num_shown3 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_accident),value=5,key='slider3')
    st.dataframe(df_accident[:num_shown3])

    st.write('### - Visualize and Remove Outliers')
    fig1 = px.box(df_accident,x=['Age'],
                hover_data=['Survived'],title='Age',width=700)
    st.write(fig1)
    fig3 = px.box(df_accident,x=['Speed_of_Impact'],
            hover_data=['Survived'],title='Speed_of_Impact',width=700)
    st.write(fig3)
    with st.echo():
        df_accident = df_accident[df_accident['Age'] < 0.729]
        df_accident = df_accident[df_accident['Speed_of_Impact'] < 0.834]
    num_shown4 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_accident),value=5,key='slider4')
    st.dataframe(df_accident[:num_shown4])
    st.write('---')
    
    st.write('## Step 2: Model Training and Evaluation')
    st.write('### - Artificial Neural Network (ANN)')
    st.write('**1. Import Libraries for Neural Network**')
    with st.echo():
        import keras
        import tensorflow as tf
        from keras import regularizers
        import sklearn.model_selection as model_selection

    st.write('**2. Split Dataset with 60% for Training and 40% for Testing**')
    with st.echo():
        x = df_accident.drop(columns=['Survived'])
        y = df_accident['Survived']
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,train_size=0.60,random_state=999)
        
    st.write('**3. Create and Train Neural Network Model**')
    st.code( """
        model.add(keras.Input(shape=(5,)))
        model.add(keras.layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])""")
    
    st.write('**4. Evaluate Neural Network Model**')
    st.code("""
        history_model = model.fit(x_train,y_train,epochs = 80,validation_data=(x_test, y_test))
        eval = model.evaluate(x_test,y_test)""")
    save_model = tf.keras.models.load_model('./model/neural.keras')
    eval_result = save_model.evaluate(x_test, y_test)
    cols = st.columns(2)
    with cols[0]:
        st.write(f"#### **Accuracy :** {eval_result[1]*100:.5f} %")
    with cols[1]:
        st.write(f"#### **Loss :** {eval_result[0]*100:.5f} %")
    st.write('---')
    # st.write('### Number of Survived: ',df_accident[df_accident['Survived'] == 1].shape[0],'/',df_accident.shape[0])
    st.write('### Let\'s click the Model in the sidebar to discover your fate!')

if page_selected == 'Model':
    st.markdown('#### **Input your data for predict**')
    cols = st.columns(2)
    with cols[0]:
        age = st.number_input('Age',min_value=1,value=50)
        # speed_impact = st.number_input('Speed of Impact (km/hr) (2 digits)',min_value=0.0,step=.01,format='%.2f')
        speed_impact = st.number_input('Speed of Impact (km/hr)',min_value=1,value=60)
        seatbelt = st.selectbox('Seatbelt Used',('Yes','No'))
        
    with cols[1]:
        sex = st.selectbox('Sex',('Male','Female'))
        helmet = st.selectbox('Helmet Used',('Yes','No'))
    btn1 = st.button('Predict!')

    if btn1:
        import tensorflow as tf
        import pickle
        loaded_model = tf.keras.models.load_model('./model/neural.keras')
        with open('./model/neural_scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        x_new = {'Age':[age],
                'Gender': [sex],
                'Speed_of_Impact':[speed_impact],
                'Helmet_Used':[helmet],
                'Seatbelt_Used':[seatbelt]}
        df_x_new = pd.DataFrame(x_new)
        df_x_new['Gender'] = df_x_new['Gender'].map({'Male': 0,'Female': 1})
        df_x_new['Seatbelt_Used'] = df_x_new['Seatbelt_Used'].map({'No': 0,'Yes': 1})
        df_x_new['Helmet_Used'] = df_x_new['Helmet_Used'].map({'No': 0,'Yes': 1})
    
        scalers = loaded_scaler
        df_x_new['Age'] = scalers['Age'].transform(df_x_new[['Age']])
        df_x_new['Speed_of_Impact'] = scalers['Speed_of_Impact'].transform(df_x_new[['Speed_of_Impact']])
        # st.write(df_x_new)

        prediction = (loaded_model.predict(df_x_new)*100)
        if (prediction > 50):
            st.markdown('#### Your chance of survival from the Road Accident is')
            st.markdown(
                f"""<p style="font-size: 26px; color: green;font-weight: bold;">
                {prediction[0, 0]:.5f} %</p>"""
                , unsafe_allow_html=True)
        else:
            st.markdown('#### Your chance of survival from the Titanic disaster is')
            st.markdown(
                f"""<p style="font-size: 26px; color: red;font-weight: bold;">
                {prediction[0, 0]:.5f} %</p>"""
                , unsafe_allow_html=True)
        st.markdown(
            """<p style="font-size: 20px;color: red;">
            <strong> Disclaimer : </strong> The survival predictions are based on data 
            from the Road Accident Survival Dataset and may not be applicable to other 
            situations. Use the results for entertainment purposes only</p>""",
            unsafe_allow_html=True)



# # Template
# page1 = option_menu("Main Menu", ["Home", "Page 1", "Page 2"], 
#                    icons=['house', 'file-earmark', 'file-earmark'],
#                    menu_icon="cast", default_index=0,orientation = 'horizontal', 
#                    styles={"container": {"padding": "5px", "background-color": "#f0f2f6"}})
# page2 = option_menu("Main Menu", ["Home", "Page 1", "Page 2"])   
# st.write(f"You selected: {page_selected}")
# # 1. as sidebar menu
# with st.sidebar:
#     selected = option_menu("Main Menu", ["Home", 'Settings'], 
#         icons=['house', 'gear'], menu_icon="cast", default_index=1)
#     selected
# # 2. horizontal menu
# selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")
# selected2
# # 3. CSS style definitions
# selected3 = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal",
#     styles={
#         "container": {"padding": "0!important", "background-color": "#fafafa"},
#         "icon": {"color": "orange", "font-size": "25px"}, 
#         "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#         "nav-link-selected": {"background-color": "green"},
#     }
# )
# # 4. Manual item selection
# if st.session_state.get('switch_button', False):
#     st.session_state['menu_option'] = (st.session_state.get('menu_option', 0) + 1) % 4
#     manual_select = st.session_state['menu_option']
# else:
#     manual_select = None
# selected4 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     orientation="horizontal", manual_select=manual_select, key='menu_4')
# st.button(f"Move to Next {st.session_state.get('menu_option', 1)}", key='switch_button')
# selected4
# # 5. Add on_change callback
# def on_change(key):
#     selection = st.session_state[key]
#     st.write(f"Selection changed to {selection}") 
# selected5 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'],
#                         icons=['house', 'cloud-upload', "list-task", 'gear'],
#                         on_change=on_change, key='menu_5', orientation="horizontal")
# selected5