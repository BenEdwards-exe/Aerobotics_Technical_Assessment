FROM python:3.12-slim

# Set the working directry
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the orchard module
COPY app/ ./app
COPY orchard/ ./orchard
# COPY .env ./

# Expose the port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app/app.py"]
