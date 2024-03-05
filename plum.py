import streamlit as st
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import easyocr
from PIL import Image
import pandas as pd
import numpy as np
import re
import mysql.connector
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
import easyocr
from joblib import load
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
import spacy
from spacy import displacy
import plotly.express as px
import plotly.graph_objects as go

#python -m streamlit run plum.py
data = pd.read_csv("upd_plum.csv")
   
selected = option_menu(None, ["Home","EDA","Questions"],
                       icons=None,
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "25px", "text-align": "centre", "margin": "-3px",
                                            "--hover-color": "#545454"},
                               "icon": {"font-size": "30px"},
                               "container": {"max-width": "5000px"},
                               "nav-link-selected": {"background-color": "#ff5757"}})

if selected == "Home":
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open("plum_io_logo.jpg"), width=300)
        with col2:
            st.markdown("## :black[**Introduction :**] This dataset provides information about the performance of different  groups in customer success teams at Plum.Based on my observation the metrics in dataset look like time series analysis.")
            st.markdown("## :black[**The original dataset contained some null values. These were handled by replacing/treating them, without compromising the size of the dataset.**]")
            st.markdown("## :green[**Technologies Used :**] web (Streamlit) , business intelligence (Power BI) applications,Streamlit, Pandas")


if selected == "EDA":
    # Function to load data
    def load_data(file_path):
        try:
            st.write(f"Loading data from: {file_path}")
            data = pd.read_csv(file_path)
            st.success("Data loaded successfully.")
            return data
        except Exception as e:
            st.error(f"Error loading the file: {str(e)}")
            return None

# Function to perform EDA
    def perform_eda(data):
        st.title("Exploratory Data Analysis (EDA)")

        # Display the first few rows of the dataset
        st.subheader("Preview of the Data")
        st.dataframe(data.head())

        # Summary statistics
        st.subheader("Summary Statistics")
        st.table(data.describe())

        # Data Information
        st.subheader("Data Information")
        st.table(data.info())

        # Univariate Analysis
        st.subheader("Univariate Analysis")

        # Histogram
        selected_column_hist = st.selectbox("Select a column for histogram:", data.columns)
        fig_hist = px.histogram(data, x=selected_column_hist, title=f'Histogram of {selected_column_hist}')
        st.plotly_chart(fig_hist)

        # Boxplot
        selected_column_box = st.selectbox("Select a column for boxplot:", data.columns)
        fig_box = px.box(data, x=selected_column_box, title=f'Boxplot of {selected_column_box}')
        st.plotly_chart(fig_box)

        # Bivariate Analysis
        st.subheader("Bivariate Analysis")

        # Scatter plot
        x_axis = st.selectbox("Select X-axis for scatter plot:", data.columns)
        y_axis = st.selectbox("Select Y-axis for scatter plot:", data.columns)
        fig_scatter = px.scatter(data, x=x_axis, y=y_axis, title=f'Scatter Plot: {x_axis} vs {y_axis}')
        st.plotly_chart(fig_scatter)

        # Correlation matrix
        st.subheader("Correlation Matrix")
        fig_corr = px.imshow(data.corr(), color_continuous_scale='viridis', title='Correlation Matrix')
        st.plotly_chart(fig_corr)

        # Multivariate Analysis
        st.subheader("Multivariate Analysis")

        # Pair plot
        fig_pair = px.scatter_matrix(data, title='Pair Plot')
        st.plotly_chart(fig_pair)

# Main function
    def main():
        st.title("EDA with Streamlit")

        st.sidebar.title("EDA & Report")

        # Upload dataset
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            data = load_data(uploaded_file)

            if data is not None:
                perform_eda(data)

    if __name__ == "__main__":
        main() 


if selected == "Questions":

    st.write("## :orange[Select any question to get Insights]")
    questions = st.selectbox('Questions',
                             ['Click the question that you would like to query',
                                "1. Which groups are quick, slow Etcc?",
                                "2. What type of tickets are taking the most time to resolve?",
                                "3. Which group have completed task as per the status?",
                                "4. which group have replied fast to customer query as the code shows below?",
                                "5. Which team is completing the task on time?"
                              ])
    if questions == "1. Which groups are quick, slow Etcc?":
        st.image(Image.open("ques2.png"), width=750)
        st.markdown("## :green[**Visualization Used :**] Ploty") 
        st.write()


    elif questions == "2. What type of tickets are taking the most time to resolve?":
        st.image(Image.open("pil.png"),width=750)
        st.markdown("## :green[**Visualization Used :**] Ploty") 
        st.write()

    elif questions == "3. Which group have completed task as per the status?":
        st.write("Analyzing task completion by group and status...")

        result = data.groupby(['Group', 'Status']).size().unstack(fill_value=0).reset_index()

        st.table(result)
    elif questions == "4. which group have replied fast to customer query as the code shows below?":
        st.write("The below table shows which group have replied fast to customer query")

        reply_by_group = data.groupby('Group')['Replies'].max().reset_index()

        st.write(reply_by_group)

    elif questions == "5. Which team is completing the task on time?":
        st.write("which team completing task on time as shown below")

        timely = data.groupby(['Group','Resolution time']).size().unstack(fill_value=0).reset_index()

        st.write(timely)