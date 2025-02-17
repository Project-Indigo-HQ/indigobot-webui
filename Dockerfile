# Use Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/indigobot /app/indigobot

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose the ports for Chainlit
EXPOSE 8000  

WORKDIR /app/indigobot

CMD ["chainlit", "run", "src/indigobot/cl.py", "--host", "0.0.0.0", "--port", "8000"]

