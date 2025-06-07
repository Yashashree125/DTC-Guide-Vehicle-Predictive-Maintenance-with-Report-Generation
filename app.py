import streamlit as st
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils.report_utils import generate_report, get_live_prediction, format_engine_parameters
from utils.scale_utils import scale_engine_parameters
from utils.faiss_utils import query_faiss_index, query_model_manuals


# Load resources
df = pd.read_excel("data/DTC_Info.xlsx")

model = SentenceTransformer('all-MiniLM-L6-v2')

xgb_model_path = "models/classification_model.pkl"
with open(xgb_model_path, "rb") as file:
    xgb_model = pickle.load(file)

# FAISS Index and Text Data
citroen_index = faiss.read_index("embeddings/citroen.index")
peugeot_index = faiss.read_index("embeddings/peugeot.index")
volkswagen_index = faiss.read_index("embeddings/volkswagen.index")

citroen_texts = open("embeddings/citroen_texts.txt", encoding='utf-8').read().splitlines()
peugeot_texts = open("embeddings/peugeot_texts.txt", encoding='utf-8').read().splitlines()
volkswagen_texts = open("embeddings/volkswagen_texts.txt", encoding='utf-8').read().splitlines()

feature_names = [
    "ENGINE_RPM", "ENGINE_COOLANT_TEMP", "ENGINE_LOAD", "THROTTLE_POS",
    "INTAKE_MANIFOLD_PRESSURE", "AIR_INTAKE_TEMP", "ENGINE_POWER", "TIMING_ADVANCE"
]

# Sidebar input form
st.sidebar.header("Input Parameters")
engine_inputs = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, step=0.1)
    engine_inputs.append(value)

model_name = st.sidebar.selectbox("Car Model", ["Citroen", "Peugeot", "Volkswagen"])

api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

generate_button = st.sidebar.button("Generate Report")

# Main UI for report display
st.title("Car Diagnostic Report Generator")

if generate_button:
    
    if not api_key:
        st.error("Please enter a valid OpenAI API key to continue.")
    else:
        scaled_values = scale_engine_parameters(engine_inputs)
        prediction_int, prediction_label = get_live_prediction(xgb_model, scaled_values)

        engine_values = format_engine_parameters(engine_inputs)
        
        st.write(f"### Predicted DTC Code: {prediction_label}")
        dtc_code = prediction_label

        # Querying FAISS index
        if model_name == "Citroen":
            model_index, model_texts = citroen_index, citroen_texts
        elif model_name == "Peugeot":
            model_index, model_texts = peugeot_index, peugeot_texts
        else:
            model_index, model_texts = volkswagen_index, volkswagen_texts

        matching_texts = query_model_manuals(engine_values, model, model_index, model_texts, top_k=20)
        manual_sections = "\n".join(matching_texts)

        dtc_query_text = df.loc[df["DTC_Number"] == dtc_code, ["Code_Meaning", "Code_Description"]].values.flatten()
        dtc_info = " ".join(map(str, dtc_query_text))

        # Generate report using GPT-4o
        report = generate_report(api_key, model_name, dtc_code, engine_values, dtc_info, manual_sections)
        
        st.write("### Generated Diagnostic Report")
        st.markdown(report, unsafe_allow_html=True)

        # Download the report
        st.download_button(
            label="Download Report",
            data=report,
            file_name="diagnostic_report.txt",
            mime="text/plain"
        )