
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app, get_active_model

client = TestClient(app)

def test_health_check_no_model():
    """Test health check behavior when no model is loaded"""
    # Mock get_active_model to raise exception as it would if file missing
    with patch('api.main.get_active_model', side_effect=Exception("No model")):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_status"] == "unhealthy"

def test_health_check_with_mock_model():
    """Test health check with a mocked model"""
    mock_model = MagicMock()
    mock_model.return_value = None # Forward pass returns nothing
    
    with patch('api.main.get_active_model', return_value=mock_model):
        with patch('api.main.model_manager.get_active_version', return_value="v17.test"):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            # If DB is healthy (which it should be with sqlite default), status is healthy
            # We assume DB is healthy for this test or mock it too
            assert data["model_version"] == "v17.test"

def test_batch_scan_endpoint():
    """Test batch scan endpoint inputs"""
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "min_liquidity": 100000
    }
    # Mock background tasks to avoid actual execution
    with patch('api.main.BackgroundTasks.add_task'):
        response = client.post("/scan/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        if data.get("status") != "accepted":
            print(f"DEBUG: Status mismatch. Data: {data}")
        assert data["status"] == "accepted"

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_health_check_no_model()
        print("Health check (no model) passed")
        test_health_check_with_mock_model()
        print("Health check (mock model) passed")
        test_batch_scan_endpoint()
        print("Batch scan endpoint passed")
        print("ALL API TESTS PASSED")
    except Exception as e:
        print(f"API TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
