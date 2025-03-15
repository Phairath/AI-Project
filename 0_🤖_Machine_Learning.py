import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(
    page_title = 'Machine Learning',
    page_icon = '🤖',
    initial_sidebar_state = 'auto',
    layout = 'centered' 
)

with st.sidebar:
    page_selected = option_menu('Pages', ['Main', 'Model'], 
        menu_icon="list-ul", icons = ['house', 'box'], default_index=0,orientation = 'horizontal')

# page_option = st.sidebar.radio('Choose Page', ['Main','Model'])
# st.sidebar.success("Machine Learning")
st.sidebar.success('**Explore** the [Titanic Dataset](%s)' %'https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
st.sidebar.success('**Created by** [Phairath Jaradnaparatana 6604062630382 SEC.2](%s)' %'https://github.com/Phairath')
st.sidebar.success("**Powered by** [Streamlit](https://streamlit.io/)")

st.title('🚢 Titanic Survival Simulator: Test Your Fate with Data!')
st.write('#### \# Supervised learning leverages Ensemble Learning techniques to enhance model performance')
st.markdown('#### \# K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree')
st.markdown('<span style="color: red; font-size: 20px;">\****Information about the dataset and other details can be found in the sidebar**</span> ', unsafe_allow_html=True)

@st.cache_data(persist=True)
def load_data(url):
    try:
        df = pd.read_csv('./data/titanic.csv')
    except Exception:
        df = pd.read_csv(url)
    return df

if page_selected == 'Main':
    st.write('## Step 1: Exploratory Data Analysis (EDA)')
    df_titanic = load_data('https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv')
    # st.dataframe(df_titanic)
    num_shown = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_titanic),value=5,key='slider1')
    st.dataframe(df_titanic[:num_shown])

    st.write('### - Data Description') #(Focus on Abbreviated Features)')
    df_description = pd.DataFrame({
        'Variable': ['Pclass','Parch','SibSp','Embarked'],
        'Data Type' : ['Pclass','Parch','SibSp','Embarked'],
        'Feature Type' : ['Pclass','Parch','SibSp','Embarked'],
        'Definition': ['Ticket class','\# of parents / children aboard the Titanic',
                        '\# of siblings / spouses aboard the Titanic',
                        'Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)']
        
    })
    st.table(df_description)

    st.write('### - Remove irrelevant columns for model training')
    with st.echo():
        df_clean_titanic = df_titanic.drop(columns=['PassengerId','Name','Ticket','Cabin'])

    st.write('### - Transform data')
    st.markdown('**Data Encoding**')
    with st.echo():
        df_clean_titanic['Sex'] = df_clean_titanic['Sex'].map({'male': 0,'female': 1})
        df_clean_titanic['Embarked'] = df_clean_titanic['Embarked'].map({'C': 0,'Q': 1,'S': 2})
        # df_clean_titanic.loc[:, 'Sex'] = df_clean_titanic['Sex'].map({'male': 0, 'female': 1})
        # df_clean_titanic.loc[:,'Embarked'] = df_clean_titanic['Embarked'].map({'C': 0,'Q': 1,'S': 2})
    st.markdown('**Data Scaling**')
    with st.echo():
        scaler = MinMaxScaler()
        df_clean_titanic['Pclass'] = scaler.fit_transform(df_clean_titanic[['Pclass']])
        df_clean_titanic['Age'] = scaler.fit_transform(df_clean_titanic[['Age']])
        df_clean_titanic['SibSp'] = scaler.fit_transform(df_clean_titanic[['SibSp']])
        df_clean_titanic['Parch'] = scaler.fit_transform(df_clean_titanic[['Parch']])
        df_clean_titanic['Fare'] = scaler.fit_transform(df_clean_titanic[['Fare']])
        df_clean_titanic['Embarked'] = scaler.fit_transform(df_clean_titanic[['Embarked']])
        # df_clean_titanic.loc[:,'Pclass'] = scaler.fit_transform(df_clean_titanic[['Pclass']])
        # df_clean_titanic.loc[:,'Age'] = scaler.fit_transform(df_clean_titanic[['Age']])
        # df_clean_titanic.loc[:,'SibSp'] = scaler.fit_transform(df_clean_titanic[['SibSp']])
        # df_clean_titanic.loc[:,'Parch'] = scaler.fit_transform(df_clean_titanic[['Parch']])
        # df_clean_titanic.loc[:,'Fare'] = scaler.fit_transform(df_clean_titanic[['Fare']])
        # df_clean_titanic.loc[:,'Embarked'] = scaler.fit_transform(df_clean_titanic[['Embarked']])
    num_shown3 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_clean_titanic),value=5,key='slider3')
    st.dataframe(df_clean_titanic[:num_shown3])

    st.write('### - Remove Missing Values')
    with st.echo():
        df_clean_titanic = df_clean_titanic.dropna()
    num_shown2 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_clean_titanic),value=5,key='slider2')
    st.dataframe(df_clean_titanic[:num_shown2])

    st.write('### - Visualize and Remove Outliers')
    cols = st.columns(2)
    with cols[0]:
        fig1 = px.box(df_clean_titanic,x=['Age'],
                    hover_data=['Survived'],title='Age',width=700)
        st.write(fig1)
        fig3 = px.box(df_clean_titanic,x=['Parch'],
                hover_data=['Survived'],title='Parch',width=700)
        st.write(fig3)
    with cols[1]:
        fig2 = px.box(df_clean_titanic,x=['SibSp'],
                    hover_data=['Survived'],title='SibSp',width=700)
        st.write(fig2)
        fig4 = px.box(df_clean_titanic,x=['Fare'],
                    hover_data=['Survived'],title='Fare',width=700)
        st.write(fig4)
    with st.echo():
        df_clean_titanic = df_clean_titanic[df_clean_titanic['Age'] < 0.824]
        df_clean_titanic = df_clean_titanic[df_clean_titanic['SibSp'] < 0.375]
        df_clean_titanic = df_clean_titanic[df_clean_titanic['Parch'] < 0.5]
        df_clean_titanic = df_clean_titanic[df_clean_titanic['Fare'] < 0.138]
        # df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['Age'] >= 0.824].index,inplace=True)
        # df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['SibSp'] >= 0.375].index,inplace=True)
        # df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['Parch'] >= 0.5].index,inplace=True)
        # df_clean_titanic.drop(df_clean_titanic[df_clean_titanic['Fare'] >= 0.138].index,inplace=True)

    # q1 = df_clean_titanic['Age'].quantile(0.25)
    # q3 = df_clean_titanic['Age'].quantile(0.75)
    # iqr = q3-q1
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr
    # outliers = df_clean_titanic[(df_clean_titanic['Age'] < lower_bound) | (df_clean_titanic['Age'] > upper_bound)]
    # st.write(outliers)
    num_shown4 = st.slider('Slide to expand data: ',min_value=5,max_value=len(df_clean_titanic),value=5,key='slider4')
    st.dataframe(df_clean_titanic[:num_shown4])
    # st.write('### Number of Survived: ',df_clean_titanic[df_clean_titanic['Survived'] == 1].shape[0],'/',df_clean_titanic.shape[0])
    st.write('---')
    
    st.write('## Step 2: Model Training and Evaluation')
    st.write('### - K-Nearest Neighbors (KNN)')
    st.write('**1. Import Libraries for KNN Classification**')
    with st.echo():
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    st.write('**2. Split Dataset with 70% for Training and 30% for Testing**')
    with st.echo():
        x = df_clean_titanic.drop(columns=['Survived'])
        y = df_clean_titanic['Survived']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=99)
    st.write('**3. Create and Train KNN Model with neighbors=4**')
    with st.echo():
        knn = KNeighborsClassifier(n_neighbors = 4)
        knn.fit(x_train,y_train)
    st.write('**4. Evaluate KNN Model**')
    with st.echo():
        predict_knn = knn.predict(x_test)
        accuracy_knn = accuracy_score(y_test,predict_knn)
        precision_knn = precision_score(y_test, predict_knn, average='macro')
        recall_knn = recall_score(y_test, predict_knn, average="macro")
        f1_knn = f1_score(y_test, predict_knn, average="macro")
        # conf_matrix_knn = confusion_matrix(y_test, predict_knn)
    # st.write("KNN efficiency \n")
    cols = st.columns(4)
    with cols[0]:
        st.write(f"#### **Accuracy :** {accuracy_knn*100:.5f} %")
    with cols[1]:
        st.write(f"#### **Precision :** {precision_knn*100:.5f} %")
    with cols[2]:
        st.write(f"#### **Recall :** {recall_knn*100:.5f} %")
    with cols[3]:
        st.write(f"#### **F1-Score :** {f1_knn*100:.5f} %")

    st.write('\n')
    st.write('### - Support Vector Machine (SVM)')
    st.write('**1. Import Libraries for SVM Classification**')
    with st.echo():
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    st.write('**2. Split Dataset with 70% for Training and 30% for Testing**')
    with st.echo():
        x = df_clean_titanic.drop(columns=['Survived'])
        y = df_clean_titanic['Survived']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=99)
    st.write('**3. Create and Train SVM Model**')
    with st.echo():
        svm_model = SVC(kernel='linear',C=0.1,probability=True)
        svm_model.fit(x_train,y_train)
    st.write('**4. Evaluate KNN Model**')
    with st.echo():
        predict_svm = svm_model.predict(x_test)
        accuracy_svm = accuracy_score(y_test,predict_svm)
        precision_svm = precision_score(y_test, predict_svm, average='macro')
        recall_svm = recall_score(y_test, predict_svm, average="macro")
        f1_svm = f1_score(y_test, predict_svm, average="macro")
    cols = st.columns(4)
    with cols[0]:
        st.write(f"#### **Accuracy :** {accuracy_svm*100:.5f} %")
    with cols[1]:
        st.write(f"#### **Precision :** {precision_svm*100:.5f} %")
    with cols[2]:
        st.write(f"#### **Recall :** {recall_svm*100:.5f} %")
    with cols[3]:
        st.write(f"#### **F1-Score :** {f1_svm*100:.5f} %")

    st.write('\n')
    st.write('### - Decision Tree')
    st.write('**1. Import Libraries for Decision Tree Classification**')
    with st.echo():
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    st.write('**2. Split Dataset with 70% for Training and 30% for Testing**')
    with st.echo():
        x = df_clean_titanic.drop(columns=['Survived'])
        y = df_clean_titanic['Survived']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=99)
    st.write('**3. Create and Train Decision Tree Model**')
    with st.echo():
        tree_model = DecisionTreeClassifier(random_state=99)
        tree_model.fit(x_train,y_train)
    st.write('**4. Evaluate Decision Tree Model**')
    with st.echo():
        predict_tree = tree_model.predict(x_test)
        accuracy_tree = accuracy_score(y_test,predict_tree)
        precision_tree = precision_score(y_test, predict_tree, average='macro')
        recall_tree = recall_score(y_test, predict_tree, average="macro")
        f1_tree = f1_score(y_test, predict_tree, average="macro")
    cols = st.columns(4)
    with cols[0]:
        st.write(f"#### **Accuracy :** {accuracy_tree*100:.5f} %")
    with cols[1]:
        st.write(f"#### **Precision :** {precision_tree*100:.5f} %")
    with cols[2]:
        st.write(f"#### **Recall :** {recall_tree*100:.5f} %")
    with cols[3]:
        st.write(f"#### **F1-Score :** {f1_tree*100:.5f} %")

    st.write('\n')
    st.write('### - Ensemble Learning with VotingClassifier')
    with st.echo():
        from sklearn.ensemble import VotingClassifier
        x = df_clean_titanic.drop(columns=['Survived'])
        y = df_clean_titanic['Survived']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=99)
        voting_model = VotingClassifier(estimators=[
            ('decision_tree', tree_model),
            ('svm', svm_model),
            ('knn', knn)
            ], voting='soft')
        voting_model.fit(x_train, y_train)
        predict_voting = voting_model.predict(x_test)
        accuracy_voting = accuracy_score(y_test,predict_voting)
        precision_voting = precision_score(y_test, predict_voting, average='macro')
        recall_voting = recall_score(y_test, predict_voting, average="macro")
        f1_voting = f1_score(y_test, predict_voting, average="macro")
    cols = st.columns(4)
    with cols[0]:
        st.write(f"#### **Accuracy :** {accuracy_voting*100:.5f} %")
    with cols[1]:
        st.write(f"#### **Precision :** {precision_voting*100:.5f} %")
    with cols[2]:
        st.write(f"#### **Recall :** {recall_voting*100:.5f} %")
    with cols[3]:
        st.write(f"#### **F1-Score :** {f1_voting*100:.5f} %")
    st.write('---')
    st.write('### Let\'s click the Model in the sidebar to discover your fate!')

    # Test New Data
    # x_new = {'Pclass':[1],
    #                'Sex': [1],
    #                'Age':[0.645],
    #                'SibSp':[0],
    #                'Parch':[0],
    #                'Fare':[0.1],
    #                'Embarked':[1]}
    # df_x_new = pd.DataFrame(x_new)
    # st.write(knn.predict_proba(df_x_new)*100)
    # st.write(svm_model.predict_proba(df_x_new)*100)
    # st.write(tree_model.predict_proba(df_x_new)*100)
    # st.write(voting_model.predict(df_x_new))
    # st.write(voting_model.predict_proba(df_x_new)*100)
    
    # st.write('\n')
    # st.write('## Step 3: Model Visualization')
    # accuracies = [accuracy_knn*100, accuracy_svm*100, accuracy_tree*100, accuracy_voting*100]
    # labels = ['KNN', 'SVM', 'Decision Tree', 'Voting Classifier']
    # fig5 = px.bar(y=accuracies,x=labels,
    #                 title='Test',width=700)
    # st.write(fig5)

if page_selected == 'Model':
    st.markdown('#### **Input your data for predict**')
    cols = st.columns(2)
    with cols[0]:
        pClass = st.selectbox('Passenger Class',('1st class','2nd class','3rd class'))
        # st.number_input('Passenger Class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)',min_value=1,max_value=3)
        age = st.number_input('Age',min_value=1,value=35)
        parch = st.number_input('Parents/Children Aboard',min_value=0,value=1)
        embarked = st.selectbox('Port of Embarkation',('Cherbourg','Queenstown','Southampton'))
        btn1 = st.button('Predict!')
        
    with cols[1]:
        sex = st.selectbox('Sex',('Male','Female'))
        # st.number_input('Sex (1 = Male, 2 = Female)',min_value=1,max_value=2)
        sibSp = st.number_input('Siblings/Spouses Aboard',min_value=0,value=0)
        fare = st.number_input('Fare (USD)',min_value=0,value=100)

    if btn1:
        import pickle
        with open('./model/supervised(3).pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        x_new = {'Pclass':[pClass],
                   'Sex': [sex],
                   'Age':[age],
                   'SibSp':[sibSp],
                   'Parch':[parch],
                   'Fare':[fare],
                   'Embarked':[embarked]}
        df_x_new = pd.DataFrame(x_new)
        df_x_new['Pclass'] = df_x_new['Pclass'].map({'1st class': 1,'2nd class': 2,'3rd class':3})
        df_x_new['Sex'] = df_x_new['Sex'].map({'Male': 0,'Female': 1})
        df_x_new['Embarked'] = df_x_new['Embarked'].map({'Cherbourg': 0,'Queenstown': 1,'Southampton': 2})
        scalers = loaded_model['scalers']
        df_x_new['Pclass'] = scalers['Pclass'].transform(df_x_new[['Pclass']])
        df_x_new['Sex'] = scalers['Sex'].transform(df_x_new[['Sex']])
        df_x_new['Age'] = scalers['Age'].transform(df_x_new[['Age']])
        df_x_new['SibSp'] = scalers['SibSp'].transform(df_x_new[['SibSp']])
        df_x_new['Parch'] = scalers['Parch'].transform(df_x_new[['Parch']])
        df_x_new['Fare'] = scalers['Fare'].transform(df_x_new[['Fare']])
        df_x_new['Embarked'] = scalers['Embarked'].transform(df_x_new[['Embarked']])
        # st.write(df_x_new)

        prediction = (loaded_model['model'].predict_proba(df_x_new)*100)
        if (prediction[0,0] > 50):
            st.markdown(f'#### Your chance of survival from the Titanic disaster is')
            st.markdown(
                f"""<p style="font-size: 26px; color: green;font-weight: bold;">
                {prediction[0, 0]:.5f} %</p>"""
                , unsafe_allow_html=True)
        else:
            st.markdown(f'#### Your chance of survival from the Titanic disaster is')
            st.markdown(
                f"""<p style="font-size: 26px; color: red;font-weight: bold;">
                {prediction[0, 0]:.5f} %</p>"""
                , unsafe_allow_html=True)
        st.markdown(
            """<p style="font-size: 20px;color: red;">
            <strong> Disclaimer : </strong> The survival predictions are based on data from the Titanic disaster
            and may not be applicable to other situations. Use the results for entertainment purposes
            only</p>""",unsafe_allow_html=True)