# Use the official Python image.
FROM python:3.9-slim

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Expose the port that Streamlit uses.
EXPOSE 8501

# Set environment variables.
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the app using your custom script.
CMD ["python", "custom_script.py"]