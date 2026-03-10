"""
GRADIO APP
Simple web interface for sentiment analysis.
"""

import gradio as gr
from inference import predict

def analyze(text):
    return predict(text)

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=3, label="Enter text"),
    outputs=[gr.JSON(label="Prediction")],
    title="Simple Sentiment Analyzer"
)

if __name__ == "__main__":
    demo.launch()
