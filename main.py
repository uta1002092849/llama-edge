"""
FastAPI application for LLaMA Edge Prediction Model
Serves the model through a secure REST API endpoint
"""
import os
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import json
from transformers import AutoModel
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key configuration
API_KEY_NAME = "x-api-token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Global variable to store the model wrapper
model_wrapper = None


class UnifiedIdMapper:
    """Mapper class to convert between old and new IDs"""
    
    def __init__(self, nodes: dict[int, str], edges: dict[int, str]) -> None:
        # since all key in JSON are str, convert them to int
        nodes = {int(k): v for k, v in nodes.items()}
        edges = {int(k): v for k, v in edges.items()}

        self.nodes = nodes
        self.edges = edges

        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(self.nodes.keys()))}
        edge_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(edges.keys()))}
        shift = len(nodes)

        self.old_to_new: dict[int, tuple[int, bool]] = {
            **{old_id: (new_id, False) for old_id, new_id in node_mapping.items()},
            **{old_id: (new_id + shift, True) for old_id, new_id in edge_mapping.items()},
        }
        # reverse mapping: new_id -> (old_id, is_edge)
        self.new_to_old: dict[int, tuple[int, bool]] = {
            new_id: (old_id, is_edge)
            for old_id, (new_id, is_edge) in self.old_to_new.items()
        }

        # Label maps
        self.old_id_to_label: dict[int, str] = {**nodes, **edges}
        self.new_id_to_label: dict[int, str] = {
            new_id: self.old_id_to_label[old_id] for old_id, (new_id, _) in self.old_to_new.items()
        }

        self.label_to_old_ids: dict[str, list[tuple[int, bool]]] = {}
        self.label_to_new_ids: dict[str, list[tuple[int, bool]]] = {}
        for old_id, (new_id, is_edge) in self.old_to_new.items():
            label = self.old_id_to_label.get(old_id)
            if label is None:
                continue
            self.label_to_old_ids.setdefault(label, []).append((old_id, is_edge))
            self.label_to_new_ids.setdefault(label, []).append((new_id, is_edge))

    @classmethod
    def from_file(cls, mapper_path: str):
        with open(mapper_path, "r") as f:
            data = json.load(f)
            return cls(data['nodes'], data['edges'])

    def map_old_id(self, old_id: int) -> tuple[int, bool]:
        return self.old_to_new[old_id]

    def map_new_id(self, new_id: int) -> tuple[int, bool]:
        return self.new_to_old[new_id]

    def label_from_old_id(self, old_id: int) -> str:
        return self.old_id_to_label[old_id]

    def label_from_new_id(self, new_id: int) -> str:
        return self.new_id_to_label[new_id]

    def old_ids_from_label(self, label: str) -> list[tuple[int, bool]]:
        return self.label_to_old_ids.get(label, [])

    def new_ids_from_label(self, label: str) -> list[tuple[int, bool]]:
        return self.label_to_new_ids.get(label, [])


class ModelWrapper:
    """Wrapper class for the LLaMA Edge model"""
    
    def __init__(self, mapper_path: str, model, device: str = "cuda"):
        # Load Mapper
        print(f"Loading mapper from {mapper_path}...")
        self.mapper = UnifiedIdMapper.from_file(mapper_path)

        # set model
        self.model = model

        # Set device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU.")
            self.device = torch.device("cpu")
        elif device == "mps":  # Handle MPS explicitly if requested or available
            self.device = torch.device("mps")
        else:
            self.device = torch.device(device)

        print(f"Moving model to {self.device}...")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, old_ids_context: list[int]) -> list[tuple[float, int, str]]:
        """
        Args:
            old_ids_context: List of old IDs defining the context.
        Returns:
            sorted_predictions: List of (prob, old_id, label) sorted by probability descending.
        """
        # 1. Convert context list of old IDs to new IDs
        input_ids = []
        for old_id in old_ids_context:
            # We assume the input old_ids exist in the mapper
            new_id, _ = self.mapper.map_old_id(old_id)
            input_ids.append(new_id)

        # 2. Run inference
        # Create tensor on result device (batch size = 1)
        model_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(model_input)
            # Get logits for the last token in the sequence
            last_token_logits = logits[0, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)

        # 3. Sort by probability descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        sorted_probs = sorted_probs.tolist()
        sorted_indices = sorted_indices.tolist()  # These indices are the new_ids

        # 4. Create result list with mapping applied
        results = []
        for prob, new_id in zip(sorted_probs, sorted_indices):
            try:
                # map_new_id returns (old_id, is_edge)
                old_id, _ = self.mapper.map_new_id(new_id)
                label = self.mapper.label_from_new_id(new_id)
                results.append((prob, old_id, label))
            except KeyError:
                # Handle indices not in mapper (e.g., padding tokens)
                results.append((prob, -1, "<PAD/UNK>"))

        return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    global model_wrapper
    
    print("Loading LLaMA Edge model...")
    model_id = "crab27/llama3-edge"
    
    # Load the model with trust_remote_code=True
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    
    # Load the UnifiedIdMapper
    mapper_path = hf_hub_download(repo_id=model_id, filename="unified_id_mapper.json")
    
    # Initialize the wrapper (will use CUDA if available, otherwise CPU)
    model_wrapper = ModelWrapper(mapper_path, model)
    print("Model loaded successfully!")
    
    yield
    
    # Cleanup
    print("Shutting down and cleaning up...")
    model_wrapper = None


# Initialize FastAPI app
app = FastAPI(
    title="LLaMA Edge Prediction API",
    description="API for predicting edges in knowledge graphs using LLaMA Edge model",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for edge prediction"""
    context_ids: List[int] = Field(
        ...,
        description="List of old IDs defining the context for prediction",
        example=[108, 112, 117, 349, 421, 608, 761, 765, 805, 912]
    )
    top_k: Optional[int] = Field(
        10,
        description="Number of top predictions to return",
        ge=1,
        le=100
    )


class PredictionItem(BaseModel):
    """Single prediction item"""
    rank: int = Field(..., description="Rank of this prediction (1-based)")
    id: int = Field(..., description="Predicted old ID")
    label: str = Field(..., description="Label corresponding to the ID")
    probability: float = Field(..., description="Prediction probability")


class PredictionResponse(BaseModel):
    """Response model for edge prediction"""
    predictions: List[PredictionItem] = Field(
        ...,
        description="List of top-k predictions sorted by probability"
    )
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made"
    )


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify the API key from the request header"""
    expected_api_key = os.getenv("API_KEY")
    
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server"
        )
    
    if api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "service": "LLaMA Edge Prediction API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_wrapper is not None
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict edge given context",
    description="Predict the most likely edges given a list of context IDs"
)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
) -> PredictionResponse:
    """
    Predict edges based on context IDs.
    
    The model will return predictions sorted by probability in descending order.
    You can control how many predictions to return using the top_k parameter.
    """
    if model_wrapper is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Get predictions from model
        predictions = model_wrapper.predict(request.context_ids)
        
        # Limit to top_k results
        top_predictions = predictions[:request.top_k]
        
        # Format response
        prediction_items = [
            PredictionItem(
                rank=rank,
                id=pred_id,
                label=label,
                probability=prob
            )
            for rank, (prob, pred_id, label) in enumerate(top_predictions, start=1)
        ]
        
        return PredictionResponse(
            predictions=prediction_items,
            total_predictions=len(predictions)
        )
        
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid context ID in input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
