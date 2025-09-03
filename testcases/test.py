"""
Test suite for Cricket Match Prediction API
"""
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cricket_ai.api.main import app
from cricket_ai.llm.explain import _mock_explanation

client = TestClient(app)

def test_health_endpoint():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    """Test prediction with valid CSV"""
    test_data = pd.DataFrame({
        'total_runs': [50, 60],
        'wickets': [2, 3],
        'target': [150, 160],
        'balls_left': [30, 25]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_file_path = f.name
    
    try:
        with open(temp_file_path, 'rb') as f:
            response = client.post("/predict", files={"file": ("test.csv", f, "text/csv")})
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction_id" in data
        assert "predictions_file" in data
        
    finally:
        os.unlink(temp_file_path)

def test_mock_explanation():
    """Test mock explanation function"""
    context = {'total_runs': 50, 'wickets': 2, 'target': 150, 'balls_left': 30}
    explanation = _mock_explanation(context, 0.8)
    assert len(explanation) > 0
    assert isinstance(explanation, str)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
