# Import the Libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Libraries for EDA
import pandas_profiling as pp
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
import codecs
import streamlit.components.v1 as components

# Libraries for Model Validation and Test
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score , f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Mushroom Classification', page_icon='mushroom1.jpg')
                   #,layout = "wide" , initial_sidebar_state="collapsed")


if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()
if 'data_label' not in st.session_state:
    st.session_state['data_label'] = pd.DataFrame()
    
if 'start' not in st.session_state:
    st.session_state['start'] = 0



def main_page():
    st.image('mushroom33.jpg')
    st.markdown("<h1 style='text-align: center;'>Mushroom Classification</h1>", unsafe_allow_html=True)
    st.write('---')
    st.session_state['start'] = 1
    
    def change():
        st.session_state['data'] = pd.DataFrame()
        st.session_state['data_label'] = pd.DataFrame()
    
    # Input DataSet Used to Train the Model---------------------------------------------------------------------------------------------------------------------------------------

    st.sidebar.header('User Input Parameters')
    st.sidebar.write('---')
    data_file = st.sidebar.selectbox(
        label ="DataSet for Model Training",
        options=['Default','Upload'], on_change=change())
    
    # To use default file for training the model
    if data_file == 'Default':
        st.subheader('Input DataFrame')
        data = pd.read_csv('mushrooms.csv')
        st.session_state['data'] = data
        st.dataframe(data)
        use_cols = ['class','spore-print-color','gill-color','gill-size','stalk-root',
                    'habitat','stalk-shape','odor','population']
        data_label = data[use_cols]
        st.session_state['data_label'] = data_label

    # Upload another file
    if data_file == 'Upload':
        file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv', key='a')
        
        if file == None:
            st.error('Please Upload the file')
            st.stop()
            
        else:
            data = pd.read_csv(file)
            st.session_state['data'] = data
            st.subheader('Input DataSet')
            st.dataframe(data)
            
            # Columns use to Train the Model (columns which are more important, based on Feature Importance)
            use_cols = ['class','spore-print-color','gill-color','gill-size','stalk-root',
                        'habitat','stalk-shape','odor','population']
            
            # What to do if Columns is Present OR Not Present
            try:
                data_label = data[use_cols]
                st.session_state['data_label'] = data_label

            except:
                st.error('Please Upload the correct file, your file must contain below columns')
                st.write(pd.DataFrame(use_cols, columns=['columns']))
                st.stop()

    # Input DataSet is Taken------------------------------------------------------------------------------------------------------------------------------------------------------


def eda():
#    import pandas_profiling as pp
#    from streamlit_pandas_profiling import st_profile_report

#    import sweetviz as sv
#    import codecs
#    import streamlit.components.v1 as components
    
    st.image('mushroom21.jpg', 'Mushroom Classification')
    st.header('**EDA**')
    st.write('---')
    
    try:
        if 'pp_eda_report' not in st.session_state:
            st.session_state['pp_eda_report'] = None
        if 'sw_eda_report' not in st.session_state:
            st.session_state['sw_eda_report'] = None
            

        eda = st.selectbox('EDA', ['Pandas Profiling', 'Sweetviz'])
        
        # EDA Process-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Pandas Profiling============================================================================================================================================
            
        if eda == 'Pandas Profiling':
            if st.session_state['pp_eda_report'] is None:
                #st.write('pp_eda_report is None')
                EDA_report= pp.ProfileReport(st.session_state['data'], title="Pandas Profiling Report", explorative=True, dark_mode=True)
                st.session_state['pp_eda_report'] = EDA_report
            
            st.subheader("Pandas Profiling EDA Report")
            st_profile_report(st.session_state['pp_eda_report'])
    
        # Pandas Profiling End========================================================================================================================================
    
    
    
        # Sweetviz====================================================================================================================================================
        def st_display_sweetviz(report_html):
            report_file = codecs.open(report_html, 'r')
            page = report_file.read()
            return page
            
            
        if eda == 'Sweetviz':
            report = sv.analyze(st.session_state['data'])
            report.show_html('report.html', open_browser=False)
            if st.session_state['sw_eda_report'] == None:
                #st.write('sw_eda_report is None')
                page = st_display_sweetviz('report.html')
                st.session_state['sw_eda_report']=page
            
            st.subheader("Sweetviz EDA Report")
            components.html(st.session_state['sw_eda_report'], width=900, height=800, scrolling=True)
    
        # Sweetviz End================================================================================================================================================
    
    
    
    
        # Train & Test Data Comparision===============================================================================================================================
    
        # Label Encoder
        label = LabelEncoder()
    
        # Encoded DataFrame
        final_data = st.session_state['data_label'].apply(label.fit_transform)
    
    
        # Spliting into X, y
        X_label = final_data.iloc[:,1:]
        y_label = final_data.iloc[:,0]
    
#        from sklearn.model_selection import train_test_split
        # split X and y into training and testing sets
        Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_label, y_label, 
                                                                test_size = 0.2, random_state = 0)
    
        compare_report = sv.compare([Xl_train, 'Train'], [Xl_test, 'Test'])
        compare_report.show_html('compare.html', open_browser=False)
    
        if st.button('Generate Comparison Report b/w Train & Test'):
            page_compare = st_display_sweetviz('compare.html')
            
            st.subheader("Comparison Report b/w Train & Test DataSet")
            components.html(page_compare, width=900, height=800, scrolling=True)
    
    
        # Train & Test Data Comparision End===========================================================================================================================
        
        
    except:
        st.error('Go to **Main Page** and Upload the DataSet')




def model_validation():
    try:
        # Import the Libraries
#        import matplotlib.pyplot as plt
#        import seaborn as sns
        #import numpy as np
    
    
        st.image('mushroom41.jpg', 'Mushroom Classification')
        st.header('**Model Validation**')
        st.write('---')
    
    
        data_label = st.session_state['data_label']
        
    
        # Data Preprocessing i.e. Label Encoding--------------------------------------------------------------------------------------------------------------------------------------
    
    
        #from sklearn.preprocessing import LabelEncoder
    
        # Label Encoder
        label = LabelEncoder()
    
        # Encoded DataFrame
        final_data = data_label.apply(label.fit_transform)
        st.subheader('Encoded DataSet use for Train and Test')
        st.dataframe(final_data)
    
        # Data Encoding End-----------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Model Building--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Import necessary Libraries
#        from sklearn.tree import  DecisionTreeClassifier
#        from sklearn.model_selection import train_test_split
#        from sklearn.metrics import classification_report , accuracy_score , f1_score, confusion_matrix
                
                
        # Spliting into X, y
        X_label = final_data.iloc[:,1:]
        y_label = final_data.iloc[:,0]
    
        # split X and y into training and testing sets
        Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_label, y_label, 
                                                                test_size = 0.2, random_state = 0)
    
        # Model Training
        tre = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=2, random_state=0)
        tre.fit(Xl_train, yl_train)
    
        # Model Building End----------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Model Validation------------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Training Validation_______________________________________________________________________________________
    
        #Predict for X dataset
        y_train_predict_tree = tre.predict(Xl_train)
    
        # Training Accuracy and F1-score
        train_acc_score = accuracy_score(yl_train, y_train_predict_tree)
        train_f1_score = f1_score(yl_train, y_train_predict_tree)
    
        st.subheader('Training Accuracy')
        st.write('Train Accuracy Score : ' , train_acc_score)
        st.write('Train F1 Score : ' , train_f1_score)
    
        # print classification report
        st.text('Model Report on Training DataSet:\n ' 
                +classification_report(yl_train, y_train_predict_tree, digits=4))
    
    
        # Confusion Matrix for Train Data
        cm = pd.DataFrame(confusion_matrix(yl_train,y_train_predict_tree), 
                                           columns=['Edible', 'Poisonous'], index=['Edible', 'Poisonous'])
    
        sns.set_theme(style='dark')
        sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})
    
        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
        #ax.tick_params(grid_color='r', labelcolor='r', color='r')
            
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.figure.axes[-1].tick_params(colors='white')
        plt.show()
        st.write(fig)
    
        st.text("")
        st.text("")
        st.write('#')
    
        # Training Validation End___________________________________________________________________________________
    
                
        # Testing Validation________________________________________________________________________________________
    
        #Predict for X dataset       
        y_test_predict_tree = tre.predict(Xl_test)
    
        # Testing Accuracy and F1-Score
        test_acc_score = accuracy_score(yl_test, y_test_predict_tree)
        test_f1_score = f1_score(yl_test, y_test_predict_tree)
        st.subheader('Testing Accuracy')
        st.write('Test Accuracy Score : ' , test_acc_score)
        st.write('Test F1 Score : ' , test_f1_score)
    
        # print classification report
        st.text('Model Report on Testing DataSet:\n ' 
                +classification_report(yl_test, y_test_predict_tree, digits=4))       
                
    
        # Confusion Matrix for Test Data      
        cm = pd.DataFrame(confusion_matrix(yl_test,y_test_predict_tree), 
                                           columns=['Edible', 'Poisonous'], index=['Edible', 'Poisonous'])
    
        sns.set_theme(style='dark')
        sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})
    
        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
        #ax.tick_params(grid_color='r', labelcolor='r', color='r')
            
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.figure.axes[-1].tick_params(colors='white')
        plt.show()
        st.write(fig)        
    
        # Test Validation End_______________________________________________________________________________________
                
        # Model Validation End-------------------------------------------------------------------------------------------------------------------------------------------------------------
                
            
    except:
        st.error('Go to **Main Page** and Upload the DataSet')
            
            
            
            
            
            

def model_test():
    try:
        st.image('mushroom31.jpg', 'Mushroom Classification')
        st.header('**Model Prediction**')
        st.write('---')
    
        # Input DataSet Used to Train the Model-----------------------------------------------------------------------------------------------------------------------------------------
        
        data = st.session_state['data']
        
        
        # File to be Classify-----------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Upload that file you want to be Classify
        st.sidebar.write('**Upload the Data you want to Classify**')
        test_file = st.sidebar.file_uploader('Upload DataSet In "csv" formate', type = 'csv', key='b')
    
        # If File is not Uploaded
        if test_file == None:
            st.error('Please Upload the file')
            st.stop()
            
        # If File is Uploaded
        else:
            test_data = pd.read_csv(test_file)
            st.subheader('DataSet to be Classify')
            st.dataframe(test_data)
    
        # Column that are used for Test DataSet
        use_test_cols = ['spore-print-color','gill-color','gill-size','stalk-root',
                    'habitat','stalk-shape','odor','population']
    
        # What to do if Columns is Present OR Not Present
        try:
            test_data = test_data[use_test_cols]
    
        except:
            st.error('File to be Classify is not correct, Please upload the correct file and your file contain below columns')
            st.write(pd.DataFrame(use_test_cols, columns=['columns']))
            st.stop()
            
        # File to be Classify End-------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Model Building and Training---------------------------------------------------------------------------------------------------------------------------------------------------
    
#        from sklearn.tree import  DecisionTreeClassifier
    
        # Columns use to Train the Model
        use_cols = ['class','spore-print-color','gill-color','gill-size','stalk-root',
                    'habitat','stalk-shape','odor','population']
    
        # Final Training DataFrame
        data_label = data[use_cols]
    
        # Label Encoding
        label = LabelEncoder()
        final_data = data_label.apply(label.fit_transform)
    
        # Spliting into X, y
        X_label = final_data.iloc[:,1:]
        y_label = final_data.iloc[:,0]
    
        # Model Training
        tre = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=2, random_state=0)
        tre.fit(X_label, y_label)
    
        # Model Building End------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    
        # Final Prediction--------------------------------------------------------------------------------------------------------------------------------------------------------------
    
        # Dictionary use for Mapping the Test Data besed on Train Data
        dt_class = {0:'Edible', 1:'Poisonous'}
        dt_spore_print_color = {'k': 2, 'n': 3, 'u': 6, 'h': 1, 'w': 7, 'r': 5, 'o': 4, 'y': 8, 'b': 0}
        dt_gill_color = {'k': 4, 'n': 5, 'g': 2, 'p': 7, 'w': 10, 'h': 3, 'u': 9, 'e': 1, 'b': 0, 'r': 8, 'y': 11, 'o': 6}
        dt_gill_size = {'n': 1, 'b': 0}
        dt_stalk_root = {'e': 3, 'c': 2, 'b': 1, 'r': 4, '?': 0}
        dt_habitat = {'u': 5, 'g': 1, 'm': 3, 'd': 0, 'p': 4, 'w': 6, 'l': 2}
        dt_stalk_shape = {'e': 0, 't': 1}
        dt_odor = {'p': 6, 'a': 0, 'l': 3, 'n': 5, 'f': 2, 'c': 1, 'y': 8, 's': 7, 'm': 4}
        dt_population = {'s': 3, 'n': 2, 'a': 0, 'v': 4, 'y': 5, 'c': 1}
    
    
        # Single Dictionary with Key name as Column name
        map_label = {'spore-print-color':dt_spore_print_color, 'gill-color':dt_gill_color, 'gill-size':dt_gill_size, 
                     'stalk-root':dt_stalk_root, 'habitat':dt_habitat, 'stalk-shape':dt_stalk_shape, 'odor':dt_odor, 
                     'population':dt_population, 'class':dt_class}
    
    
        # Label Encoding done with "map" command and map_label dictionary
        test_label = pd.DataFrame()
        for col in use_test_cols:
            test_label[col] = test_data[col].map(map_label[col])
    
        #st.dataframe(test_label)
    
        # Test Prediction
        y_pred_tree = tre.predict(test_label)
    
        # Prediction DataFrame with Test Input
        pred_data = test_data.copy()
        pred_data['class'] = y_pred_tree
        pred_data['class'] = pred_data['class'].map({0:'Edible', 1:'Poisonous'})
    
        # Fuction define to color the dataframe
        def color_df(clas):
            if clas == 'Poisonous':
                color = 'tomato'
            elif clas == 'Edible':
                color = 'green'
            else:
                color = 'dimgrey'
                
            return f'background-color: {color}'
    
    
        # Final DataFrame
        st.subheader('Classified Data or Output')
        st.dataframe(pred_data.style.applymap(color_df, subset=['class']))
    
    
    
        # Value Counts of Final Dataframe
        dt = {'Mushroom_Classification':pred_data['class'].value_counts().index.tolist(), 
              'Counts':pred_data['class'].value_counts().values.tolist()}
        value_counts = pd.DataFrame(dt)
        st.subheader('Value Counts of Classified Data')
        st.dataframe(value_counts.style.applymap(color_df, subset=['Mushroom_Classification']))
    
        # Final Pridiction Over---------------------------------------------------------------------------------------------------------------------------------------------------------

    except:
        st.warning('Or Go to **Main Page** and Upload the DataSet')


def made_by():
    st.header('**Made By**')
    st.write('---')
    col1, col2, col3= st.columns([3,6,4])


    with col1:
        st.subheader("**Name**")
        st.write('Ayush Patidar')
        st.write('Anup Vetal')
        st.write('Farzan Nawaz')
        st.write('Prashant Khandekar')
        st.write('Prasad Waje')

    with col2:
        st.subheader("**Mail**")
        st.write('ayushpatidar1712@gmail.com')
        st.write('anupsv1997@gmail.com')
        st.write('farzannawaz4787@gmail.com')
        st.write('pkhandekar108@gmail.com')
        st.write('prasadwaje2029@gmail.com')

    with col3:
        st.subheader("**Mob. No.**")
        st.write('9131985346')
        st.write('8668314822')
        st.write('7898480467')
        st.write('7030870449')
        st.write('8999714455')


page_names_to_funcs = {
    "Main Page": main_page,
    "EDA": eda,
    "Model Validation": model_validation,
    "Data Classification": model_test,
    "Made By": made_by
}

st.sidebar.header('Select a Task')
st.sidebar.write('---')
selected_page = st.sidebar.selectbox("Select", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

