# Use Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY requirements.txt .
COPY pyproject.toml .
COPY src/ ./src/
COPY .chainlit/ ./.chainlit/
COPY chainlit.md .
COPY public/ ./public/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose the port for Chainlit
EXPOSE 8080

CMD ["chainlit", "run", "src/indigobot/clui.py", "--host", "0.0.0.0", "--port", "8080"]

