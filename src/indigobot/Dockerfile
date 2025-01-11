# Base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy application files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "__main__.py"]
