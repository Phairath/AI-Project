import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
st.set_page_config(
    page_title="Neural Network",
    page_icon="🧠",
    initial_sidebar_state = 'auto',
    layout = 'centered'
    # menu_items = {
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)
with st.sidebar:
    page_selected = option_menu('Pages', ['Main', 'Model'], 
        menu_icon="list-ul", icons = ['house', 'box'], default_index=0,orientation = 'horizontal')

# st.sidebar.success("Neural Network")
st.sidebar.success('**Explore** a [Titanic Dataset](%s)' %'https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
st.sidebar.success('**Created by** Phairath Jaradnaparatana 6604062630382 SEC.2')
st.sidebar.success("**Powered by** [Streamlit](https://streamlit.io/)")

# page1 = option_menu("Main Menu", ["Home", "Page 1", "Page 2"], 
#                    icons=['house', 'file-earmark', 'file-earmark'],
#                    menu_icon="cast", default_index=0,orientation = 'horizontal', 
#                    styles={"container": {"padding": "5px", "background-color": "#f0f2f6"}})
# page2 = option_menu("Main Menu", ["Home", "Page 1", "Page 2"])   
# st.write(f"You selected: {page_selected}")




# import streamlit as st
# from streamlit_option_menu import option_menu

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