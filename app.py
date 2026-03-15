"""
GRADIO APP
Web interface for the Sentiment Analysis model.
This script acts as the 'Frontend' of the project.
"""

import gradio as gr
from inference import predict

def analyze_sentiment(text):
    """
    Bridge function between the UI and the Inference logic.
    """
    if not text.strip():
        return "Please enter some text."
    
    # Using the fast predict function from our previous script
    results = predict(text)
    
    # Formatting output for better UI display
    label = results["label"].upper()
    confidence = results["confidence"]
    return f"SENTIMENT: {label}\nCONFIDENCE: {confidence:.2%}"

# Create a polished UI
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Type a tweet here...", 
        label="Social Media Post"
    ),
    outputs=gr.Textbox(label="Analysis Result"),
    title="MachineInnovators Inc. - Reputation Monitor",
    description="Analyze social media sentiment in real-time. Data is logged for monitoring.",
    examples=[
        ["I love the new update!"],
        ["The service was terrible and slow."],
        ["The package arrived this morning."]
    ]
)

if __name__ == "__main__":
    demo.launch()
