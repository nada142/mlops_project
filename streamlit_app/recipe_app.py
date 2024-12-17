<<<<<<< HEAD
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.express as px

# Function to normalize data (MinMaxScaler)
def normalize_data(data, scaler):
    if not hasattr(scaler, 'scale_'):
        scaler.fit(data)
    return pd.DataFrame(scaler.transform(data), columns=data.columns)

# Load the MinMaxScaler if available, or train a new one if necessary
try:
    with open("streamlit_app/minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = MinMaxScaler()
    with open("streamlit_app/minmax_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Load the complexity model
with open("streamlit_app/model.pkl", "rb") as f:
    complexity_model = pickle.load(f)

# Function to predict recipe complexity
def predict_complexity(recipe_text):
    features = np.random.rand(1, 3)
    features = np.hstack([features, np.array([[len(recipe_text.split())]])])
    normalized_features = normalize_data(pd.DataFrame(features), scaler)
    return complexity_model.predict(normalized_features)[0]

# Streamlit UI
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Project Overview", "Recipe Complexity Predictor", "Dashboard"])

if page == "Project Overview":
    st.title("Project Overview")
    st.subheader("Objective")
    st.write("""
        The objective of this project is to predict the complexity level of a recipe based on 
        its ingredients, steps, and cooking time, while providing useful insights via a dashboard.
    """)
    
    st.subheader("Context")
    st.write("""
        Using machine learning to predict recipe complexity and provide an interactive dataset analysis.
    """)
    
    st.subheader("Data Description")
    st.write("""
        The data consists of recipes with details like ingredients, preparation steps.
        Features:
        * id (int): ID.
        * title (str): Title of the recipe.
        * ingredients (list of str): Ingredients.
        * directions (list of str): Instruction steps.
        * link (str): URL link.
        * NER (list of str): NER food entities.
    """)

elif page == "Recipe Complexity Predictor":
    st.title("Recipe Complexity Predictor")

    st.header("📊 Recipe Complexity Predictor")
    recipe_text = st.text_area("Paste a recipe to analyze its complexity:")
    if st.button("Predict Complexity"):
        with st.spinner("Analyzing complexity..."):
            try:
                complexity = predict_complexity(recipe_text)
                complexity_map = {0: "Easy", 1: "Moderate", 2: "Complex"}
                st.success(f"The predicted complexity of the recipe is: **{complexity_map[complexity]}**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif page == "Dashboard":
    st.title("📊 Recipe Dataset Dashboard")
    
    # Key Highlights (KPIs)
    st.subheader("Dataset Highlights")
    col1, col2, col3 = st.columns(3)
    data = pd.read_csv("streamlit_app/data_processed.csv")  # Replace `dataset_path` with the actual path to your dataset.

    try:
        # Compute KPIs
        total_recipes = len(data)
        avg_ingredients = data['num_ingredients'].mean()  # Corrected: Use `data` DataFrame
        avg_steps = data['num_steps'].mean()  # Corrected: Use `data` DataFrame

        # Display KPIs
        col1.metric("📋 Total Recipes", f"{total_recipes:,}")
        col2.metric("🥗 Avg. Ingredients/Recipe", f"{avg_ingredients:.2f}")
        col3.metric("📝 Avg. Steps/Recipe", f"{avg_steps:.2f}")
    except KeyError as e:
        st.error(f"Missing data for key highlights: {e}")

    st.subheader("Ingredient Analysis")
    ingredient_counts = data['ingredients'].apply(lambda x: len(x.split(',')))
    st.bar_chart(ingredient_counts.value_counts().sort_index())
    # Step Count Distribution
    st.subheader("Step Count Distribution")
    try:
        fig_steps = px.histogram(
            data,
            x='num_steps',
            nbins=20,
            title="Step Count Distribution",
            labels={'num_steps': 'Number of Steps'}
        )
        st.plotly_chart(fig_steps)
    except KeyError:
        st.warning("The dataset does not have a 'num_steps' column.")
=======
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.express as px

# Function to normalize data (MinMaxScaler)
def normalize_data(data, scaler):
    if not hasattr(scaler, 'scale_'):
        scaler.fit(data)
    return pd.DataFrame(scaler.transform(data), columns=data.columns)

# Load the MinMaxScaler if available, or train a new one if necessary
try:
    with open("streamlit_app/minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = MinMaxScaler()
    with open("streamlit_app/minmax_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Load the complexity model
with open("streamlit_app/model.pkl", "rb") as f:
    complexity_model = pickle.load(f)

# Function to predict recipe complexity
def predict_complexity(recipe_text):
    features = np.random.rand(1, 3)
    features = np.hstack([features, np.array([[len(recipe_text.split())]])])
    normalized_features = normalize_data(pd.DataFrame(features), scaler)
    return complexity_model.predict(normalized_features)[0]

# Streamlit UI
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Project Overview", "Recipe Complexity Predictor", "Dashboard"])

if page == "Project Overview":
    st.title("Project Overview")
    st.subheader("Objective")
    st.write("""
        The objective of this project is to predict the complexity level of a recipe based on 
        its ingredients, steps, and cooking time, while providing useful insights via a dashboard.
    """)
    
    st.subheader("Context")
    st.write("""
        Using machine learning to predict recipe complexity and provide an interactive dataset analysis.
    """)
    
    st.subheader("Data Description")
    st.write("""
        The data consists of recipes with details like ingredients, preparation steps.
        Features:
        * id (int): ID.
        * title (str): Title of the recipe.
        * ingredients (list of str): Ingredients.
        * directions (list of str): Instruction steps.
        * link (str): URL link.
        * NER (list of str): NER food entities.
    """)

elif page == "Recipe Complexity Predictor":
    st.title("Recipe Complexity Predictor")

    st.header("📊 Recipe Complexity Predictor")
    recipe_text = st.text_area("Paste a recipe to analyze its complexity:")
    if st.button("Predict Complexity"):
        with st.spinner("Analyzing complexity..."):
            try:
                complexity = predict_complexity(recipe_text)
                complexity_map = {0: "Easy", 1: "Moderate", 2: "Complex"}
                st.success(f"The predicted complexity of the recipe is: **{complexity_map[complexity]}**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif page == "Dashboard":
    st.title("📊 Recipe Dataset Dashboard")
    
    # Key Highlights (KPIs)
    st.subheader("Dataset Highlights")
    col1, col2, col3 = st.columns(3)
    data = pd.read_csv("streamlit_app/data_processed.csv")  # Replace `dataset_path` with the actual path to your dataset.

    try:
        # Compute KPIs
        total_recipes = len(data)
        avg_ingredients = data['num_ingredients'].mean()  # Corrected: Use `data` DataFrame
        avg_steps = data['num_steps'].mean()  # Corrected: Use `data` DataFrame

        # Display KPIs
        col1.metric("📋 Total Recipes", f"{total_recipes:,}")
        col2.metric("🥗 Avg. Ingredients/Recipe", f"{avg_ingredients:.2f}")
        col3.metric("📝 Avg. Steps/Recipe", f"{avg_steps:.2f}")
    except KeyError as e:
        st.error(f"Missing data for key highlights: {e}")

    st.subheader("Ingredient Analysis")
    ingredient_counts = data['ingredients'].apply(lambda x: len(x.split(',')))
    st.bar_chart(ingredient_counts.value_counts().sort_index())
    # Step Count Distribution
    st.subheader("Step Count Distribution")
    try:
        fig_steps = px.histogram(
            data,
            x='num_steps',
            nbins=20,
            title="Step Count Distribution",
            labels={'num_steps': 'Number of Steps'}
        )
        st.plotly_chart(fig_steps)
    except KeyError:
        st.warning("The dataset does not have a 'num_steps' column.")
>>>>>>> 29474ea (Initial commit)
