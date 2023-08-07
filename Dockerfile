# Use an official Python runtime as a base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Remove unnecessary locale data to avoid read-only file system issue
RUN rm -rf /usr/share/locale/*


# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the requirements.txt file first to leverage Docker caching
COPY requirements.txt .

# Install any Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Install python-multipart
RUN pip install --no-cache-dir python-multipart

# Copy the model file and Python script to the container

COPY app /app
#COPY karim_work.py .
#COPY main1.py .
#COPY karim_model.h5 .

# Set the command to run the FastAPI server when the container starts
CMD ["uvicorn", "main1:app", "--host", "0.0.0.0", "--port", "8000"]
