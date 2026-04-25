import gradio as gr
from model_api import predict_voltage
import tempfile
import shutil
import os

def run_prediction(formula, cif_file):
    if cif_file is None:
        return "Please upload CIF file"

    temp_path = tempfile.mktemp(suffix=".cif")
    shutil.copy(cif_file.name, temp_path)

    try:
        voltage = predict_voltage(temp_path, formula)
        return f"🔋 Predicted Voltage: {round(voltage,3)} V"
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=run_prediction,
    inputs=[
        gr.Textbox(label="Chemical Formula", placeholder="LiCoO2"),
        gr.File(label="Upload CIF File")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="⚡ Battery Voltage Predictor",
    description="Hybrid Tabular Graph Neural Network based voltage prediction tool."
)

demo.launch()
