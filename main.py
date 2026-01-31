"""
FastAPI application for LLaMA Edge Prediction Model
Serves the model through a secure REST API endpoint
"""
import os
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import json
from transformers import AutoModel, AutoConfig
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

    def predict(self, old_ids_context: list[int], candidate_old_ids: list[int]) -> list[tuple[int, float]]:
        """
        Args:
            old_ids_context: List of old IDs defining the context (query_session).
            candidate_old_ids: List of candidate old IDs to rank.
        Returns:
            sorted_predictions: List of (old_id, score) sorted by score descending.
        """
        # 1. Convert context list of old IDs to new IDs
        input_ids = []
        for old_id in old_ids_context:
            # We assume the input old_ids exist in the mapper
            new_id, _ = self.mapper.map_old_id(old_id)
            input_ids.append(new_id)

        # 2. Convert candidate old IDs to new IDs
        candidate_new_ids = []
        for old_id in candidate_old_ids:
            new_id, _ = self.mapper.map_old_id(old_id)
            candidate_new_ids.append(new_id)

        # 3. Run inference
        # Create tensor on result device (batch size = 1)
        model_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(model_input)
            # Get logits for the last token in the sequence
            last_token_logits = logits[0, -1, :]
            
            # 4. Gather logits only for candidate edges
            candidate_logits = last_token_logits[candidate_new_ids]
            
            # 5. Renormalize: compute softmax over only the candidate logits
            candidate_probs = torch.softmax(candidate_logits, dim=-1)

        # 6. Create list of (old_id, score) pairs
        results = []
        for i, old_id in enumerate(candidate_old_ids):
            score = candidate_probs[i].item()
            results.append((old_id, score))
        
        # 7. Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

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
        dtype=torch.bfloat16
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
    query_session: List[int] = Field(
        ...,
        description="List of old IDs defining the context/query session",
        json_schema_extra={"example": [108, 112, 117, 349, 421, 608, 761, 765, 805, 912]}
    )
    candidate_edges: List[int] = Field(
        ...,
        description="List of candidate edge old IDs to rank",
        json_schema_extra={"example": [100, 200, 300, 400, 500]}
    )


class PredictionResponse(BaseModel):
    """Response model for edge prediction"""
    ranked_edges: List[List] = Field(
        ...,
        description="List of [candidate_edge_id, score] pairs sorted by score descending",
        json_schema_extra={"example": [[300, 0.45], [100, 0.30], [500, 0.15], [200, 0.08], [400, 0.02]]}
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
    summary="Rank candidate edges given query session",
    description="Rank candidate edges based on query session context"
)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
) -> PredictionResponse:
    """
    Rank candidate edges based on query session.
    
    The model will compute probabilities only over the provided candidate edges
    and return them sorted by score in descending order.
    """
    if model_wrapper is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Get predictions from model (returns list of (old_id, score) tuples)
        predictions = model_wrapper.predict(
            request.query_session,
            request.candidate_edges
        )
        
        # Format response as list of [id, score] pairs
        ranked_edges = [[old_id, score] for old_id, score in predictions]
        
        return PredictionResponse(
            ranked_edges=ranked_edges
        )
        
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ID in input: {str(e)}"
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
