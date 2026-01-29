# LLaMA Edge Prediction

A FastAPI-based REST API for serving the LLaMA Edge model for knowledge graph edge prediction.

## Features

- 🚀 FastAPI framework with OpenAPI documentation
- 🔐 API key authentication via `x-api-token` header
- 🐳 Docker containerization support
- 📊 Automatic model loading on startup
- 🏥 Health check endpoints
- 📝 Interactive API documentation (Swagger UI & ReDoc)

## Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerization)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd llama-edge
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Edit `.env` and set your API key:
```
API_KEY=your-secure-api-key-here
PORT=8000
```

## Running the Application

### Local Development

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker

1. Build the Docker image:
```bash
docker build -t llama-edge-api .
```

2. Run the container:
```bash
docker run -d \
  -p 8000:8000 \
  -e API_KEY=your-secure-api-key-here \
  --name llama-edge-api \
  llama-edge-api
```

For GPU support (CUDA):
```bash
docker run -d \
  -p 8000:8000 \
  -e API_KEY=your-secure-api-key-here \
  --gpus all \
  --name llama-edge-api \
  llama-edge-api
```

## API Documentation

Once the application is running, access the interactive documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage

### Authentication

All requests (except `/` and `/health`) require an API key in the header:

```
x-api-token: your-api-key-here
```

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-token: your-api-key-here" \
  -d '{
    "context_ids": [108, 112, 117, 349, 421, 608, 761, 765, 805, 912, 930, 937, 940, 1076, 1095, 1125, 1133, 1188, 1510, 1948, 1958, 47178924],
    "top_k": 10
  }'
```

### Request Format

```json
{
  "context_ids": [108, 112, 117, ...],
  "top_k": 10
}
```

- `context_ids`: List of old IDs defining the context for prediction (required)
- `top_k`: Number of top predictions to return (optional, default: 10, max: 100)

### Response Format

```json
{
  "predictions": [
    {
      "rank": 1,
      "id": 47185647,
      "label": "/location/mailing_address/state_province_region-/location/mailing_address/citytown",
      "probability": 0.892341
    },
    ...
  ],
  "total_predictions": 9942
}
```

## Example Usage with Python

```python
import requests

url = "http://localhost:8000/predict"
headers = {
    "x-api-token": "your-api-key-here",
    "Content-Type": "application/json"
}
data = {
    "context_ids": [108, 112, 117, 349, 421, 608, 761, 765, 805, 912],
    "top_k": 10
}

response = requests.post(url, json=data, headers=headers)
predictions = response.json()
print(predictions)
```

## Security Notes

1. **Never commit your `.env` file** - it contains sensitive API keys
2. Use strong, randomly generated API keys in production
3. Consider using HTTPS in production environments
4. The API key is verified on every protected request
5. Rate limiting is recommended for production deployments

## Model Information

- **Model**: crab27/llama3-edge
- **Purpose**: Knowledge graph edge prediction
- **Total Classes**: 9942 (nodes + edges)
- **Framework**: PyTorch + Transformers

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]