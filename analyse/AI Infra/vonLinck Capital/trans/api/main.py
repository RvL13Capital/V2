"""
TRANS Production API - FIXED VERSION
=====================================

Fixed implementation with all TODOs resolved.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
import asyncio
import time
import json
import uuid
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import pandas as pd

# Import TRANS components
from core.pattern_scanner import ConsolidationPatternScanner
from core.pattern_detector import TemporalPatternDetector
from core.exceptions import ValidationError, TemporalConsistencyError, DataIntegrityError
from models.temporal_hybrid_v18 import HybridFeatureNetwork as HybridTemporalClassifier
from models.model_manager import ModelManager
from database.connection import db_manager, get_db_session
from database.models import Pattern, Prediction, ModelVersion, SystemLog, MetricSnapshot, TaskStatus
from utils.logging_config import get_production_logger_manager, get_production_logger
from utils.error_handler import get_error_handler, CircuitBreaker
from utils.monitoring import MetricsCollector
from config import (
    FEATURE_DIM,  # Number of temporal features (14)
    NUM_CLASSES,
    calculate_expected_value  # 3-class system
)

# Initialize components
app = FastAPI(
    title="TRANS Production API",
    description="Temporal Pattern Detection and Prediction System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize logging and monitoring
logger_manager = get_production_logger_manager()
logger = get_production_logger(__name__, "api")
error_handler = get_error_handler()
metrics_collector = MetricsCollector()

# Initialize circuit breakers
data_breaker = CircuitBreaker(
    name="data_loading",
    failure_threshold=5,
    timeout=60
)
model_breaker = CircuitBreaker(
    name="model_inference",
    failure_threshold=3,
    timeout=30
)

# Global model manager
model_manager = ModelManager()

# Task tracking store (in production, use Redis or database)
task_store: Dict[str, Dict] = {}

# Process pool for parallel scanning
executor = ProcessPoolExecutor(max_workers=4)


# =====================================================================
# Middleware for Response Time Tracking
# =====================================================================

@app.middleware("http")
async def track_response_time(request, call_next):
    """Track API response times for monitoring."""
    start_time = time.time()
    response = await call_next(request)
    response_time = (time.time() - start_time) * 1000  # Convert to ms

    # Record metric
    metrics_collector.record_response_time(request.url.path, response_time)

    # Add header
    response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
    return response


# =====================================================================
# Dependency Injection
# =====================================================================

@lru_cache()
def get_pattern_scanner() -> ConsolidationPatternScanner:
    """Get cached pattern scanner instance."""
    return ConsolidationPatternScanner()


@lru_cache()
def get_pattern_detector() -> TemporalPatternDetector:
    """Get cached pattern detector instance.

    Uses INFERENCE mode to include active/qualifying patterns (not just terminal).
    This allows predictions on unripe patterns with placeholder labels.
    """
    from config.constants import ProcessingMode
    return TemporalPatternDetector(mode=ProcessingMode.INFERENCE)


def get_active_model() -> HybridTemporalClassifier:
    """Get current active model."""
    model = model_manager.get_active_model()
    if not model:
        # Load default model if none active
        model_path = Path("output/models/best_model.pt")
        if model_path.exists():
            model = HybridTemporalClassifier(
                input_dim=FEATURE_DIM,
                num_classes=NUM_CLASSES
            )
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
            model.eval()
            model_manager.register_model(
                model=model,
                version="v17.1",
                architecture="hybrid"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="No trained model available"
            )
    return model


# =====================================================================
# Pydantic Models (same as before, adding TaskStatus models)
# =====================================================================

class BatchScanRequest(BaseModel):
    """Request model for batch pattern scanning."""
    tickers: List[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    min_liquidity: Optional[float] = 50000  # Default MIN_LIQUIDITY_DOLLAR
    parallel_workers: int = Field(default=4, ge=1, le=10)

class PredictionRequest(BaseModel):
    """Request model for pattern prediction."""
    pattern_id: str
    model_version: Optional[str] = None
    include_features: bool = False

class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# =====================================================================
# Health Check with Real Model Status
# =====================================================================

@app.get("/health")
async def health_check():
    """Enhanced health check with actual model status."""
    start_time = time.time()

    # Check database
    db_health = db_manager.health_check()

    # Check model health
    model_health = "healthy"
    model_version = None
    try:
        model = get_active_model()
        # Test inference
        test_input = torch.randn(1, 20, FEATURE_DIM)
        with torch.no_grad():
            _ = model(test_input)
        model_version = model_manager.get_active_version()
    except Exception as e:
        model_health = "unhealthy"
        logger.error(f"Model health check failed: {e}")

    # Get pattern counts
    with get_db_session() as session:
        active_patterns = session.query(Pattern).filter(
            Pattern.status == "ACTIVE"
        ).count()

        pending_predictions = session.query(Pattern).filter(
            Pattern.status == "LABELED",
            Pattern.outcome_class.isnot(None)
        ).count()

    # Calculate metrics
    recent_errors = len([e for e in error_handler.error_history
                       if (datetime.utcnow() - e["timestamp"]).total_seconds() < 3600])

    # Get average response time from metrics
    avg_response_time = metrics_collector.get_average_response_time(window_minutes=60)

    response_time = (time.time() - start_time) * 1000

    return {
        "status": "healthy" if db_health["status"] == "healthy" and model_health == "healthy" else "degraded",
        "database_status": db_health["status"],
        "model_status": model_health,
        "model_version": model_version,
        "active_patterns": active_patterns,
        "pending_predictions": pending_predictions,
        "error_rate_1h": recent_errors,
        "api_response_time_ms": avg_response_time,
        "health_check_time_ms": response_time,
        "timestamp": datetime.utcnow()
    }


# =====================================================================
# Pattern Scanning with Parallel Processing
# =====================================================================

async def _scan_ticker_async(
    ticker: str,
    start_date: Optional[date],
    end_date: Optional[date],
    scanner: ConsolidationPatternScanner
) -> Dict:
    """Async wrapper for ticker scanning."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        scanner.scan_ticker,
        ticker,
        start_date,
        end_date
    )


@app.post("/scan/batch")
async def scan_batch(
    request: BatchScanRequest,
    background_tasks: BackgroundTasks
):
    """Enhanced batch scanning with parallel processing."""
    task_id = str(uuid.uuid4())

    # Initialize task in store
    task_store[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "started_at": datetime.utcnow(),
        "total": len(request.tickers),
        "completed": 0,
        "results": []
    }

    # Save to database
    with get_db_session() as session:
        db_task = TaskStatus(
            task_id=task_id,
            task_type="batch_scan",
            status="pending",
            parameters={
                "tickers": request.tickers,
                "total_items": len(request.tickers),
                "completed_items": 0
            }
        )
        session.add(db_task)
        session.commit()

    # Add background task
    background_tasks.add_task(
        _process_batch_scan_parallel,
        task_id,
        request.tickers,
        request.start_date,
        request.end_date,
        request.parallel_workers
    )

    return {
        "task_id": task_id,
        "status": "accepted",
        "message": f"Scanning {len(request.tickers)} tickers in background with {request.parallel_workers} workers"
    }


async def _process_batch_scan_parallel(
    task_id: str,
    tickers: List[str],
    start_date: Optional[date],
    end_date: Optional[date],
    workers: int
):
    """Process batch scan with parallel workers."""
    scanner = ConsolidationPatternScanner()
    task_store[task_id]["status"] = "processing"

    try:
        # Create scanning tasks
        tasks = []
        for ticker in tickers:
            task = _scan_ticker_async(ticker, start_date, end_date, scanner)
            tasks.append((ticker, task))

        # Process in batches based on worker count
        results = []
        batch_size = workers

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(
                *[task for _, task in batch],
                return_exceptions=True
            )

            # Process results
            for (ticker, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to scan {ticker}: {result}")
                    results.append({"ticker": ticker, "error": str(result)})
                else:
                    results.append({
                        "ticker": ticker,
                        "patterns_found": result.patterns_found,
                        "processing_time_ms": result.processing_time_ms
                    })

            # Update progress
            completed = min(i + batch_size, len(tickers))
            task_store[task_id]["completed"] = completed
            task_store[task_id]["progress"] = completed / len(tickers)

            # Update database
            with get_db_session() as session:
                session.query(TaskStatus).filter(
                    TaskStatus.task_id == task_id
                ).update({
                    "status": "processing",
                    "completed_items": completed,
                    "updated_at": datetime.utcnow()
                })
                session.commit()

        # Complete task
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["completed_at"] = datetime.utcnow()
        task_store[task_id]["results"] = results

        # Update database
        with get_db_session() as session:
            session.query(TaskStatus).filter(
                TaskStatus.task_id == task_id
            ).update({
                "status": "completed",
                "completed_items": len(tickers),
                "results": json.dumps(results),
                "completed_at": datetime.utcnow()
            })
            session.commit()

        logger.info(f"Completed batch scan {task_id} for {len(tickers)} tickers")

    except Exception as e:
        logger.error(f"Batch scan {task_id} failed: {e}")
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)

        with get_db_session() as session:
            session.query(TaskStatus).filter(
                TaskStatus.task_id == task_id
            ).update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow()
            })
            session.commit()


# =====================================================================
# Prediction with Actual Model Inference
# =====================================================================

@app.post("/predict")
async def predict_pattern(request: PredictionRequest):
    """Enhanced prediction with comprehensive error handling."""
    request_start_time = time.time()

    with logger_manager.request_context() as request_id:
        logger.info(
            "Prediction requested",
            request_id=request_id,
            pattern_id=request.pattern_id,
            model_version=request.model_version
        )

        try:
            # Circuit breaker for model inference
            if model_breaker.state == "open":
                raise HTTPException(
                    status_code=503,
                    detail="Model inference temporarily unavailable (circuit breaker open)"
                )

            # Get pattern from database
            with get_db_session() as session:
                pattern = session.query(Pattern).filter(
                    Pattern.pattern_id == request.pattern_id
                ).first()

                if not pattern:
                    logger.warning(f"Pattern not found: {request.pattern_id}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Pattern {request.pattern_id} not found"
                    )

                # Validate pattern has required data
                if not pattern.ticker:
                    raise HTTPException(
                        status_code=400,
                        detail="Pattern missing ticker information"
                    )

                if not pattern.start_date or not pattern.end_date:
                    raise HTTPException(
                        status_code=400,
                        detail="Pattern missing date information"
                    )

            # Load pattern sequences with enhanced error handling
            detector = get_pattern_detector()

            try:
                sequences = detector.generate_sequences_for_pattern(
                    ticker=pattern.ticker,
                    pattern_start=pattern.start_date,
                    pattern_end=pattern.end_date
                )
            except ValidationError as e:
                logger.warning(f"Validation failed for pattern {request.pattern_id}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid pattern data: {str(e)}"
                )
            except TemporalConsistencyError as e:
                logger.error(f"Temporal integrity error for {request.pattern_id}: {e}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Data quality issue: {str(e)}"
                )
            except DataIntegrityError as e:
                logger.error(f"Data integrity error for {request.pattern_id}: {e}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Data integrity issue: {str(e)}"
                )
            except Exception as e:
                logger.error(f"Unexpected error generating sequences: {e}", exc_info=True)
                error_handler.handle_error(e, {
                    "operation": "generate_sequences",
                    "pattern_id": request.pattern_id,
                    "ticker": pattern.ticker
                })
                raise HTTPException(
                    status_code=500,
                    detail="Internal error generating sequences"
                )

            if sequences is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No sequences generated (pattern may be too short or data unavailable)"
                )

            if len(sequences) == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Pattern generated 0 sequences (data insufficient)"
                )

            logger.info(f"Generated {len(sequences)} sequences for prediction")

            # Get model
            model = get_active_model()
            model_version = model_manager.get_active_version()

            # Prepare input
            input_tensor = torch.FloatTensor(sequences[-1:])  # Use latest sequence

            # Run inference
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=-1).numpy()[0]

            # Create class probabilities dict
            class_probs = {str(i): float(probs[i]) for i in range(NUM_CLASSES)}

            # Calculate expected value
            ev = calculate_expected_value(class_probs)

            # Determine signal strength
            if ev >= 5.0:
                signal = "STRONG"
            elif ev >= 3.0:
                signal = "GOOD"
            elif ev >= 1.0:
                signal = "MODERATE"
            elif ev >= 0:
                signal = "WEAK"
            else:
                signal = "AVOID"

            # Get predicted class
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            # Extract features if requested
            features = None
            if request.include_features:
                # Get attention weights and important features
                importance = model.get_feature_importance(input_tensor)
                features = {
                    "temporal_importance": importance["temporal_importance"][0].tolist(),
                    "feature_importance": importance.get("feature_importance", {})
                }

            # Save prediction to database
            with get_db_session() as session:
                # Get actual model version ID
                model_obj = session.query(ModelVersion).filter(
                    ModelVersion.version == model_version,
                    ModelVersion.is_active == True
                ).first()

                model_version_id = model_obj.id if model_obj else 1

                db_prediction = Prediction(
                    pattern_id=request.pattern_id,
                    model_version_id=model_version_id,
                    predicted_class=predicted_class,
                    class_probabilities=class_probs,
                    expected_value=ev,
                    confidence=confidence,
                    signal_strength=signal
                )
                session.add(db_prediction)
                session.commit()

            # Record success metrics
            response_time = (time.time() - request_start_time) * 1000
            metrics_collector.record_response_time("/predict", response_time)
            logger.info(f"Prediction completed in {response_time:.2f}ms")

            return {
                "pattern_id": request.pattern_id,
                "predicted_class": predicted_class,
                "class_probabilities": class_probs,
                "expected_value": ev,
                "signal_strength": signal,
                "confidence": confidence,
                "model_version": model_version,
                "features": features,
                "timestamp": datetime.utcnow()
            }

        except HTTPException:
            # Re-raise HTTP exceptions without modification
            raise

        except Exception as e:
            # Handle unexpected errors
            logger.error(
                "Prediction failed with unexpected error",
                request_id=request_id,
                pattern_id=request.pattern_id,
                error=str(e),
                exc_info=True
            )

            # Record failure in circuit breaker
            model_breaker.record_failure()

            # Handle error with recovery framework
            error_handler.handle_error(e, {
                "operation": "predict",
                "pattern_id": request.pattern_id,
                "request_id": request_id
            })

            # Return user-friendly error
            raise HTTPException(
                status_code=500,
                detail="Prediction failed due to internal error"
            )


# =====================================================================
# Task Management with Actual Tracking
# =====================================================================

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get actual task status from store/database."""
    # Check in-memory store first
    if task_id in task_store:
        task = task_store[task_id]
        return TaskStatusResponse(
            task_id=task_id,
            status=task["status"],
            progress=task.get("progress", 0),
            result=task.get("results"),
            error=task.get("error"),
            started_at=task.get("started_at"),
            completed_at=task.get("completed_at")
        )

    # Check database
    with get_db_session() as session:
        db_task = session.query(TaskStatus).filter(
            TaskStatus.task_id == task_id
        ).first()

        if not db_task:
            raise HTTPException(status_code=404, detail="Task not found")

        return TaskStatusResponse(
            task_id=task_id,
            status=db_task.status,
            progress=db_task.completed_items / db_task.total_items if db_task.total_items > 0 else 0,
            result=json.loads(db_task.results) if db_task.results else None,
            error=db_task.error,
            started_at=db_task.created_at,
            completed_at=db_task.completed_at
        )


# =====================================================================
# Single Ticker Scan
# =====================================================================

@app.post("/scan/single")
async def scan_single_ticker(
    ticker: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    scanner: ConsolidationPatternScanner = Depends(get_pattern_scanner)
):
    """Scan a single ticker for consolidation patterns."""
    try:
        result = scanner.scan_ticker(ticker, start_date, end_date)

        # Save patterns to database
        with get_db_session() as session:
            for pattern_data in result.patterns:
                db_pattern = Pattern(
                    pattern_id=str(uuid.uuid4()),
                    ticker=ticker,
                    start_date=pattern_data['start_date'],
                    end_date=pattern_data['end_date'],
                    status=pattern_data.get('status', 'COMPLETED'),
                    metadata=pattern_data
                )
                session.add(db_pattern)
            session.commit()

        return {
            "ticker": ticker,
            "patterns_found": result.patterns_found,
            "processing_time_ms": result.processing_time_ms,
            "patterns": result.patterns,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Failed to scan {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# Pattern Retrieval and Management
# =====================================================================

@app.get("/patterns")
async def list_patterns(
    ticker: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """List patterns with optional filtering."""
    with get_db_session() as session:
        query = session.query(Pattern)

        if ticker:
            query = query.filter(Pattern.ticker == ticker)
        if status:
            query = query.filter(Pattern.status == status)

        total = query.count()
        patterns = query.order_by(Pattern.start_date.desc()).offset(offset).limit(limit).all()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "ticker": p.ticker,
                    "start_date": p.start_date,
                    "end_date": p.end_date,
                    "status": p.status,
                    "outcome_class": p.outcome_class,
                    "outcome_return": p.outcome_return,
                    "created_at": p.created_at
                }
                for p in patterns
            ]
        }


@app.get("/patterns/{pattern_id}")
async def get_pattern_details(pattern_id: str):
    """Get detailed information about a specific pattern."""
    with get_db_session() as session:
        pattern = session.query(Pattern).filter(Pattern.pattern_id == pattern_id).first()

        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")

        # Get associated predictions
        predictions = session.query(Prediction).filter(
            Prediction.pattern_id == pattern_id
        ).order_by(Prediction.created_at.desc()).all()

        return {
            "pattern_id": pattern.pattern_id,
            "ticker": pattern.ticker,
            "start_date": pattern.start_date,
            "end_date": pattern.end_date,
            "status": pattern.status,
            "outcome_class": pattern.outcome_class,
            "outcome_return": pattern.outcome_return,
            "metadata": pattern.metadata,
            "predictions": [
                {
                    "prediction_id": pred.id,
                    "predicted_class": pred.predicted_class,
                    "expected_value": pred.expected_value,
                    "signal_strength": pred.signal_strength,
                    "confidence": pred.confidence,
                    "created_at": pred.created_at
                }
                for pred in predictions
            ],
            "created_at": pattern.created_at
        }


@app.delete("/patterns/{pattern_id}")
async def delete_pattern(pattern_id: str):
    """Delete a pattern and its predictions."""
    with get_db_session() as session:
        pattern = session.query(Pattern).filter(Pattern.pattern_id == pattern_id).first()

        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")

        # Delete associated predictions
        session.query(Prediction).filter(Prediction.pattern_id == pattern_id).delete()

        # Delete pattern
        session.delete(pattern)
        session.commit()

        return {"message": f"Pattern {pattern_id} deleted successfully"}


# =====================================================================
# Predictions Retrieval
# =====================================================================

@app.get("/predictions")
async def list_predictions(
    pattern_id: Optional[str] = None,
    signal_strength: Optional[str] = None,
    min_ev: Optional[float] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """List predictions with optional filtering."""
    with get_db_session() as session:
        query = session.query(Prediction).join(Pattern)

        if pattern_id:
            query = query.filter(Prediction.pattern_id == pattern_id)
        if signal_strength:
            query = query.filter(Prediction.signal_strength == signal_strength)
        if min_ev is not None:
            query = query.filter(Prediction.expected_value >= min_ev)

        total = query.count()
        predictions = query.order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "predictions": [
                {
                    "prediction_id": pred.id,
                    "pattern_id": pred.pattern_id,
                    "ticker": pred.pattern.ticker,
                    "predicted_class": pred.predicted_class,
                    "expected_value": pred.expected_value,
                    "signal_strength": pred.signal_strength,
                    "confidence": pred.confidence,
                    "created_at": pred.created_at
                }
                for pred in predictions
            ]
        }


@app.get("/predictions/{pattern_id}")
async def get_predictions_for_pattern(pattern_id: str):
    """Get all predictions for a specific pattern."""
    with get_db_session() as session:
        predictions = session.query(Prediction).filter(
            Prediction.pattern_id == pattern_id
        ).order_by(Prediction.created_at.desc()).all()

        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions found for this pattern")

        return {
            "pattern_id": pattern_id,
            "total_predictions": len(predictions),
            "predictions": [
                {
                    "prediction_id": pred.id,
                    "predicted_class": pred.predicted_class,
                    "class_probabilities": pred.class_probabilities,
                    "expected_value": pred.expected_value,
                    "signal_strength": pred.signal_strength,
                    "confidence": pred.confidence,
                    "model_version_id": pred.model_version_id,
                    "created_at": pred.created_at
                }
                for pred in predictions
            ]
        }


# =====================================================================
# Model Management
# =====================================================================

@app.get("/models")
async def list_models():
    """List all registered model versions."""
    with get_db_session() as session:
        models = session.query(ModelVersion).order_by(
            ModelVersion.created_at.desc()
        ).all()

        return {
            "total": len(models),
            "active_version": model_manager.get_active_version(),
            "models": [
                {
                    "id": m.id,
                    "version": m.version,
                    "architecture": m.architecture,
                    "is_active": m.is_active,
                    "performance_metrics": m.performance_metrics,
                    "created_at": m.created_at
                }
                for m in models
            ]
        }


@app.post("/models/{version}/activate")
async def activate_model(version: str):
    """Activate a specific model version."""
    try:
        # Load model from disk
        model_path = Path(f"output/models/{version}_model.pt")
        if not model_path.exists():
            model_path = Path("output/models/best_model.pt")

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {version} not found")

        model = HybridTemporalClassifier(
            input_dim=FEATURE_DIM,
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()

        # Register and activate
        model_manager.register_model(model, version, "hybrid")
        model_manager.set_active_model(version)

        # Update database
        with get_db_session() as session:
            # Deactivate all models
            session.query(ModelVersion).update({"is_active": False})

            # Activate requested model
            db_model = session.query(ModelVersion).filter(
                ModelVersion.version == version
            ).first()

            if db_model:
                db_model.is_active = True
            else:
                # Create new entry
                db_model = ModelVersion(
                    version=version,
                    architecture="hybrid",
                    is_active=True
                )
                session.add(db_model)

            session.commit()

        return {
            "message": f"Model {version} activated successfully",
            "version": version,
            "architecture": "hybrid"
        }
    except Exception as e:
        logger.error(f"Failed to activate model {version}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# System Statistics and Metrics
# =====================================================================

@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics."""
    with get_db_session() as session:
        # Pattern statistics
        total_patterns = session.query(Pattern).count()
        active_patterns = session.query(Pattern).filter(Pattern.status == "ACTIVE").count()
        completed_patterns = session.query(Pattern).filter(Pattern.status == "COMPLETED").count()

        # Prediction statistics
        total_predictions = session.query(Prediction).count()
        strong_signals = session.query(Prediction).filter(
            Prediction.signal_strength == "STRONG"
        ).count()

        # Class distribution
        class_distribution = {}
        for i in range(NUM_CLASSES):
            count = session.query(Prediction).filter(
                Prediction.predicted_class == i
            ).count()
            class_distribution[f"class_{i}"] = count

        # Average metrics
        avg_ev = session.query(
            Prediction.expected_value
        ).filter(Prediction.expected_value.isnot(None)).all()

        avg_confidence = session.query(
            Prediction.confidence
        ).filter(Prediction.confidence.isnot(None)).all()

        # Model performance
        model_version = model_manager.get_active_version()

        return {
            "patterns": {
                "total": total_patterns,
                "active": active_patterns,
                "completed": completed_patterns
            },
            "predictions": {
                "total": total_predictions,
                "strong_signals": strong_signals,
                "class_distribution": class_distribution
            },
            "metrics": {
                "average_ev": float(np.mean([ev[0] for ev in avg_ev])) if avg_ev else 0,
                "average_confidence": float(np.mean([c[0] for c in avg_confidence])) if avg_confidence else 0
            },
            "system": {
                "active_model": model_version,
                "api_version": "1.0.0",
                "labeling_version": "v17"
            },
            "timestamp": datetime.utcnow()
        }


@app.get("/metrics")
async def get_metrics():
    """Get detailed system metrics from metrics collector."""
    metrics = metrics_collector.get_all_metrics()

    return {
        "response_times": metrics.get("response_times", {}),
        "error_counts": len(error_handler.error_history),
        "circuit_breakers": {
            "data_loading": {
                "state": data_breaker.state,
                "failure_count": data_breaker.failure_count,
                "last_failure_time": data_breaker.last_failure_time
            },
            "model_inference": {
                "state": model_breaker.state,
                "failure_count": model_breaker.failure_count,
                "last_failure_time": model_breaker.last_failure_time
            }
        },
        "timestamp": datetime.utcnow()
    }


# =====================================================================
# Recent Activity
# =====================================================================

@app.get("/recent")
async def get_recent_activity(
    limit: int = Query(default=20, le=100)
):
    """Get recent patterns and predictions."""
    with get_db_session() as session:
        recent_patterns = session.query(Pattern).order_by(
            Pattern.created_at.desc()
        ).limit(limit).all()

        recent_predictions = session.query(Prediction).join(Pattern).order_by(
            Prediction.created_at.desc()
        ).limit(limit).all()

        return {
            "recent_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "ticker": p.ticker,
                    "status": p.status,
                    "created_at": p.created_at
                }
                for p in recent_patterns
            ],
            "recent_predictions": [
                {
                    "prediction_id": pred.id,
                    "pattern_id": pred.pattern_id,
                    "ticker": pred.pattern.ticker,
                    "signal_strength": pred.signal_strength,
                    "expected_value": pred.expected_value,
                    "created_at": pred.created_at
                }
                for pred in recent_predictions
            ],
            "timestamp": datetime.utcnow()
        }


# =====================================================================
# Tickers Management
# =====================================================================

@app.get("/tickers")
async def list_tickers():
    """List all unique tickers in the system."""
    with get_db_session() as session:
        tickers = session.query(Pattern.ticker).distinct().all()

        ticker_stats = []
        for (ticker,) in tickers:
            pattern_count = session.query(Pattern).filter(
                Pattern.ticker == ticker
            ).count()

            latest_pattern = session.query(Pattern).filter(
                Pattern.ticker == ticker
            ).order_by(Pattern.created_at.desc()).first()

            ticker_stats.append({
                "ticker": ticker,
                "pattern_count": pattern_count,
                "latest_scan": latest_pattern.created_at if latest_pattern else None
            })

        return {
            "total_tickers": len(ticker_stats),
            "tickers": sorted(ticker_stats, key=lambda x: x["pattern_count"], reverse=True)
        }


# =====================================================================
# Backtest Trigger
# =====================================================================

@app.post("/backtest")
async def trigger_backtest(
    tickers: List[str],
    start_date: date,
    end_date: date,
    background_tasks: BackgroundTasks
):
    """Trigger a backtesting run."""
    task_id = str(uuid.uuid4())

    # Initialize task
    task_store[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "started_at": datetime.utcnow(),
        "type": "backtest",
        "metadata": {
            "tickers": tickers,
            "start_date": str(start_date),
            "end_date": str(end_date)
        }
    }

    # Save to database
    with get_db_session() as session:
        db_task = TaskStatus(
            task_id=task_id,
            task_type="backtest",
            status="pending",
            total_items=len(tickers),
            completed_items=0,
            metadata={
                "tickers": tickers,
                "start_date": str(start_date),
                "end_date": str(end_date)
            }
        )
        session.add(db_task)
        session.commit()

    background_tasks.add_task(
        _process_backtest,
        task_id,
        tickers,
        start_date,
        end_date
    )

    return {
        "task_id": task_id,
        "status": "accepted",
        "message": f"Backtest started for {len(tickers)} tickers"
    }


async def _process_backtest(
    task_id: str,
    tickers: List[str],
    start_date: date,
    end_date: date
):
    """Process backtest in background."""
    logger.info(f"Starting backtest {task_id}")
    task_store[task_id]["status"] = "processing"

    try:
        # Import backtester
        from backtesting.temporal_backtester import TemporalBacktester

        backtester = TemporalBacktester()
        results = backtester.run_backtest(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )

        task_store[task_id]["status"] = "completed"
        task_store[task_id]["completed_at"] = datetime.utcnow()
        task_store[task_id]["results"] = results

        # Update database
        with get_db_session() as session:
            session.query(TaskStatus).filter(
                TaskStatus.task_id == task_id
            ).update({
                "status": "completed",
                "completed_items": len(tickers),
                "results": json.dumps(results),
                "completed_at": datetime.utcnow()
            })
            session.commit()

        logger.info(f"Backtest {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Backtest {task_id} failed: {e}")
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)

        with get_db_session() as session:
            session.query(TaskStatus).filter(
                TaskStatus.task_id == task_id
            ).update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow()
            })
            session.commit()


# =====================================================================
# Startup and Shutdown Events
# =====================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting TRANS API server")

    # Check database connection
    if not db_manager.check_health()["status"] == "healthy":
        logger.error("Database not healthy at startup")

    # Load default model
    try:
        model = get_active_model()
        logger.info(f"Loaded model version: {model_manager.get_active_version()}")
    except Exception as e:
        logger.warning(f"No model loaded at startup: {e}")

    # Start metrics collection
    metrics_collector.start()

    logger.info("TRANS API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down TRANS API server")

    # Stop metrics collection
    metrics_collector.stop()

    # Shutdown executor
    executor.shutdown(wait=True)

    # Close database connections
    db_manager.close_all()

    logger.info("TRANS API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)