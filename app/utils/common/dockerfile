# Use official Python base image
FROM python:3.11-slim

# Copy only requirements first for caching
COPY requirements.txt .
COPY .env .env
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

CMD ["python", "run.py"]

