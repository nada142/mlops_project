
# Base Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Copy the scaler file to the correct directory
COPY minmax_scaler.pkl /app/streamlit_app/
COPY model.pkl /app/streamlit_app/
COPY data_processed.csv /app/streamlit_app/
# Install dependencies
RUN pip install --upgrade pip && pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Use JSON CMD format to avoid OS signal issues
CMD ["streamlit", "run", "recipe_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

