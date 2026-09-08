# Use an official Python runtime
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing pyc files and enable live terminal output
ENV PYTHONUNBUFFERED=1

# Step 1: Install PyTorch CPU directly using the explicit wheel URL
# (Bypasses pip's index resolver completely, preventing 8+ min resolution loops)
RUN pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl

# Step 2: Copy requirement list
COPY requirements.txt .

# Step 3: Install remaining requirements using --no-build-isolation
# (Forces pip to use the already installed CPU PyTorch instead of fetching CUDA torch from PyPI)
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy the backend code into the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run FastAPI app using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]