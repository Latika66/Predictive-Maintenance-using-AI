import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"  # fast + generous free tier

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-gemini-api-key-here":
        return None
    return genai.Client(api_key=api_key)

def generate_maintenance_report(sensor_readings, risk_score, shap_top_features):
    """
    Generates a professional maintenance report using Google's Gemini API.
    """
    client = get_gemini_client()
    if not client:
        return "⚠️ Gemini API key is missing or invalid. Please configure it in the .env file to view the AI maintenance report."

    system_prompt = """You are an expert industrial AI assistant specializing in predictive maintenance.
Your goal is to write a short, professional, plain-English maintenance report based on a machine's sensor data and its AI-generated failure risk.
The SHAP features show the most important drivers for this risk. Positive SHAP means it increased the risk, negative means it decreased.

Examples:
Input: Risk: 82%, Top Drivers: {'Torque': 0.32, 'Tool wear': 0.21, 'Rotational speed': -0.08}
Output: Machine unit shows an 82% failure risk, primarily driven by elevated torque and tool wear. This pattern is consistent with tool wear failure. Recommend inspecting the cutting tool and reducing load within the next 48 hours.

Input: Risk: 15%, Top Drivers: {'Air temperature': 0.02, 'Tool wear': 0.01}
Output: Machine unit is operating normally with a low failure risk of 15%. Sensor readings for temperature and tool wear are within acceptable parameters. No immediate maintenance is required.
"""

    user_prompt = f"Risk: {risk_score:.0%}, Top Drivers: {shap_top_features}\nSensor Readings: {sensor_readings}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.3,
                "max_output_tokens": 250,
            },
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generating report: {str(e)}"


def answer_followup_question(question, context_report, sensor_readings, risk_score, shap_top_features):
    """
    Answers user questions based on the context of the current machine.
    """
    client = get_gemini_client()
    if not client:
        return "⚠️ Gemini API key is missing or invalid."

    system_prompt = f"""You are a helpful industrial maintenance assistant. Answer the user's question specifically about the current machine using the provided context.
Do NOT hallucinate new data. If the answer is not in the context or requires external knowledge, state that you can only analyze the provided sensor data.

Context for current machine:
- Failure Risk: {risk_score:.0%}
- Top Risk Drivers (SHAP): {shap_top_features}
- Sensor Data: {sensor_readings}
- AI Maintenance Report: {context_report}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=question,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.2,
                "max_output_tokens": 300,
            },
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error answering question: {str(e)}"
