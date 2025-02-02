# Use Python 3.9 slim image
FROM python:3.9-slim
# FROM pytorch/pytorch
# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the current directory to /app in the container
COPY . .
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the necessary Python package from Hugging Face's GitHub repository
RUN pip install --no-cache-dir -r  requirements.txt
# RUN pip install --no-cache-dir git+https://github.com/huggingface/parler-tts.git
RUN pip install git+https://github.com/suno-ai/bark.git

# Set the entry point to keep the container alive (optional, remove if unnecessary)
# ENTRYPOINT ["tail", "-f", "/dev/null"]

# If there's a specific Python script to run, uncomment and modify the line below

# CMD ["python", "/app/perler_tts.py"]

# Expose port if your application uses a web service (optional)
# EXPOSE 8000
