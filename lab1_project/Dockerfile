# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
#EXPOSE 5000

# Define the command to run the application
CMD ["python", "app_lab1.py"]