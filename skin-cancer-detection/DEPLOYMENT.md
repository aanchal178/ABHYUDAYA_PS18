# Deployment Guide

## Deploying the Skin Cancer Detection Application

This guide covers different deployment options for the skin cancer detection system.

---

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Performance Optimization](#performance-optimization)

---

## Local Development Setup

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)
- 4GB+ RAM (8GB+ recommended for training)
- GPU (optional, but recommended for training)

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd skin-cancer-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify setup**
```bash
python test_setup.py
```

5. **Generate demo data (optional)**
```bash
python generate_demo_data.py
```

6. **Start the application**
```bash
python app.py
```

Visit `http://localhost:5000`

---

## Production Deployment

### Using Gunicorn (Linux/Mac)

1. **Install Gunicorn**
```bash
pip install gunicorn
```

2. **Create Gunicorn configuration** (`gunicorn_config.py`)
```python
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5
errorlog = "logs/gunicorn_error.log"
accesslog = "logs/gunicorn_access.log"
loglevel = "info"
```

3. **Run with Gunicorn**
```bash
gunicorn -c gunicorn_config.py app:app
```

### Using Nginx as Reverse Proxy

1. **Install Nginx**
```bash
sudo apt-get install nginx
```

2. **Create Nginx configuration** (`/etc/nginx/sites-available/skin-cancer-detection`)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 16M;
    }

    location /static {
        alias /path/to/skin-cancer-detection/static;
    }
}
```

3. **Enable site and restart Nginx**
```bash
sudo ln -s /etc/nginx/sites-available/skin-cancer-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs static/uploads models

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./static/uploads:/app/static/uploads
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
```

### Build and Run

```bash
# Build the image
docker build -t skin-cancer-detection .

# Run the container
docker run -d -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/static/uploads:/app/static/uploads \
  skin-cancer-detection

# Or use docker-compose
docker-compose up -d
```

---

## Cloud Deployment

### AWS EC2 Deployment

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - t2.medium or larger (for model inference)
   - Configure security group (allow HTTP/HTTPS)

2. **SSH into instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install dependencies**
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv nginx
```

4. **Clone and setup application**
```bash
git clone <repository-url>
cd skin-cancer-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. **Configure systemd service** (`/etc/systemd/system/skin-cancer.service`)
```ini
[Unit]
Description=Skin Cancer Detection App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/skin-cancer-detection
Environment="PATH=/home/ubuntu/skin-cancer-detection/venv/bin"
ExecStart=/home/ubuntu/skin-cancer-detection/venv/bin/gunicorn -c gunicorn_config.py app:app

[Install]
WantedBy=multi-user.target
```

6. **Start service**
```bash
sudo systemctl enable skin-cancer
sudo systemctl start skin-cancer
```

### Heroku Deployment

1. **Create Procfile**
```
web: gunicorn app:app
```

2. **Create runtime.txt**
```
python-3.9.16
```

3. **Deploy**
```bash
heroku login
heroku create skin-cancer-detection
git push heroku main
heroku open
```

### Google Cloud Platform (Cloud Run)

1. **Create Dockerfile** (see Docker section above)

2. **Build and push to Container Registry**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/skin-cancer-detection
```

3. **Deploy to Cloud Run**
```bash
gcloud run deploy skin-cancer-detection \
  --image gcr.io/PROJECT-ID/skin-cancer-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

---

## Performance Optimization

### Model Optimization

1. **Model Quantization**
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

2. **Use Model Caching**
```python
# In app.py, load model once at startup
model = None

def get_model():
    global model
    if model is None:
        model = load_trained_model()
    return model
```

### Application Optimization

1. **Enable Flask Caching**
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})
```

2. **Use CDN for Static Files**
- Host CSS, JS, and images on a CDN
- Update template URLs accordingly

3. **Implement Request Rate Limiting**
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ...
```

### Infrastructure Optimization

1. **Load Balancing**
   - Use multiple application instances
   - Configure load balancer (e.g., AWS ELB, GCP Load Balancer)

2. **Database for Results Storage**
   - Store prediction history
   - Use PostgreSQL or MongoDB

3. **Monitoring**
   - Set up application monitoring (e.g., New Relic, DataDog)
   - Monitor model performance metrics
   - Set up alerts for errors

---

## Security Considerations

1. **Input Validation**
   - Already implemented in app.py
   - Validate file types and sizes

2. **HTTPS**
   - Use SSL certificates (Let's Encrypt)
   - Configure Nginx for HTTPS

3. **Environment Variables**
   - Store secrets in environment variables
   - Never commit credentials to version control

4. **Regular Updates**
   - Keep dependencies updated
   - Monitor security advisories

---

## Monitoring and Maintenance

### Application Logs

```bash
# View logs
tail -f logs/training.log
tail -f logs/gunicorn_error.log
```

### Health Checks

```bash
# Check application health
curl http://localhost:5000/health
```

### Model Updates

1. Train new model with updated data
2. Evaluate performance
3. Replace `models/best_model.h5`
4. Restart application

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory during inference
- **Solution**: Reduce batch size, use model quantization, or increase RAM

**Issue**: Slow predictions
- **Solution**: Use GPU, optimize model, implement caching

**Issue**: File upload fails
- **Solution**: Check `MAX_CONTENT_LENGTH` in app.py, verify upload directory permissions

---

## Support

For deployment issues:
- Check application logs
- Review error messages
- Consult cloud provider documentation
- Check GitHub issues

---

**Team**: Dr. Homi Jehangir Bhabha  
**Project**: Skin Cancer Detection - PS 18
