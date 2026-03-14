from inference import predict 

def test_prediction_structure():
    result = predict("I love this project")
    assert isinstance(result, dict)
    assert "label" in result
    assert "scores" in result
    
    assert result["label"] in ["positive", "neutral", "negative"]