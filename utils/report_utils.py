from utils.ai_utils import create_openai_llm
from utils.scale_utils import scale_engine_parameters
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import xgboost as xgb

feature_names = [
    "ENGINE_RPM", "ENGINE_COOLANT_TEMP", "ENGINE_LOAD", "THROTTLE_POS",
    "INTAKE_MANIFOLD_PRESSURE", "AIR_INTAKE_TEMP", "ENGINE_POWER", "TIMING_ADVANCE"
]

label_to_dtc = {
    0: 'B0004', 1: 'C0300', 2: 'C1004', 3: 'P0078', 4: 'P0079',
    5: 'P0133', 6: 'P2004', 7: 'P2036', 8: 'P3000', 9: 'U1004'
}

def get_live_prediction(xgb_model, input_values):
    scaled_values = scale_engine_parameters(input_values)
    scaled_values = scaled_values.reshape(1, -1)  
    dmatrix_input = xgb.DMatrix(scaled_values, feature_names=feature_names)  
    prediction_int = int(xgb_model.predict(dmatrix_input)[0])  
    prediction_label = label_to_dtc.get(prediction_int, "Unknown")  
    return prediction_int, prediction_label

def format_engine_parameters(input_values):
    formatted_string = ", ".join([f"{feature.lower()}: {value}" for feature, value in zip(feature_names, input_values)])
    return formatted_string


# Chain-of-Thought Prompting 

# def create_llm_prompt_template():
#     template = """
#     You are an AI diagnostic assistant for cars. You are given the following information of a DTC code that is displayed with a OBD-II sensor due to certain engine parameters.
#     Your task is to generate a detailed diagnostic report based on the following information:

#     1. Engine Parameters: {engine_parameters}
#     2. Model Name: {model_name}
#     3. Diagnostic Trouble Code (DTC): {dtc_code}
    
#     DTC Query Result:
#     {dtc_query_result}

#     Relevant Manual Sections:
#     {manual_sections}
    
#     Instructions:
#     Follow these steps to generate the diagnostic report:

#     1. Understand the DTC Code:
#        - Read the DTC Query Result to understand the information, meaning, description and the reason of the DTC code.
#        - Relate it to the engine parameters of the given car model.
#        - Summarize the key points about the DTC code, to include its information,meaning, description and the possible reason of the DTC code.

#     2. Analyze the Engine Parameters:
#        - Examine the provided engine parameters.
#        - Identify any abnormalities or issues based on the engine parameters.

#     3. Review Relevant Manual Sections:
#        - Read through the Relevant Manual Sections.
#        - Extract any important information or steps related to the identified DTC code and engine parameters.You dont have to explictily mention it in the report.
#          Just remember them to give diagnosis and safety measures.

#     4. Generate Recommendations and Diagnostics:
#        - Based on the analysis of the DTC code, engine parameters, and manual sections, list detailed recommendations and diagnostic steps.
#        - Ensure these steps are actionable and easy to follow.
#        - Highlight abnormalities if any.
#        - Give proper headings for each section that a non-technical person can understand.
#        - Also include safety measures that might help in diagnosing the issue in the car and in maintaining the engine health.

#     Follow the following format for report generation:
#     ```
#     ### Diagnostic Report

#     Model Name:
#     {model_name}

#     Diagnostic Trouble Code (DTC):
#     {dtc_code}
    
#     Engine Parameters:
#     {engine_parameters}

#     ```
#     Generate the report below:
#     """
#     prompt_template = PromptTemplate(
#         input_variables=[ "model_name", "dtc_code", "engine_parameters", "dtc_query_result", "manual_sections"],
#         template=template
#     )
#     return prompt_template


def create_llm_prompt_template():
    template = """
    You are an AI diagnostic assistant for cars. Your task is to analyze a Diagnostic Trouble Code (DTC) 
    from an OBD-II sensor and generate a **detailed diagnostic report** based on the given engine 
    parameters and manual references.

    ---
    **Step 1: Understand the DTC Code**
    - Read the DTC Query Result carefully.
    - Identify the meaning, description, and likely cause of the DTC.
    - Explain the significance of this DTC and how it might affect engine performance.

    **Step 2: Analyze the Engine Parameters**
    - Examine the given engine parameters for any abnormalities.
    - Compare the values against typical operational ranges.
    - Determine if any specific parameter is contributing to the DTC.

    **Step 3: Correlate with the Manual**
    - Review the relevant manual sections.
    - Extract key diagnostic procedures, repair steps, and safety measures.
    - Understand if there are any manufacturer-specific instructions for this issue.

    **Step 4: Generate Diagnostic Insights**
    - Summarize the findings logically.
    - Provide structured recommendations in a format that is **easy to understand for non-technical users**.
    - Include **highlighted abnormalities**, suggested **next steps**, and **safety measures**.

    ---
    
    **Final Diagnostic Report Format:**
    ```
    ### Diagnostic Report

    **Model Name:** {model_name}  
    **Diagnostic Trouble Code (DTC):** {dtc_code}  
    **Engine Parameters:** {engine_parameters}  

    **DTC Analysis:**  
    (Explain the meaning, significance, and possible causes of the DTC)  

    **Engine Parameter Evaluation:**  
    (Highlight any abnormal engine readings and their impact)  

    **Manual-Based Recommendations:**  
    (Summarize key diagnostic and repair steps)  

    **Final Recommendations & Safety Measures:**  
    (Provide step-by-step actionable recommendations)  
    ```

    Now, follow these steps and generate the diagnostic report :
    """

    prompt_template = PromptTemplate(
        input_variables=["model_name", "dtc_code", "engine_parameters", "dtc_query_result", "manual_sections"],
        template=template
    )
    return prompt_template


def prepare_prompt(model_name, dtc_code, engine_parameters, dtc_query_result, manual_sections):
    prompt_data = {
        "model_name": model_name,
        "dtc_code": dtc_code,
        "engine_parameters": engine_parameters,
        "dtc_query_result": dtc_query_result,
        "manual_sections": manual_sections
    }
    return prompt_data


def generate_report(api_key, model_name, dtc_code, engine_parameters, dtc_query_result, manual_sections):
    # OpenAI LLM object
    llm = create_openai_llm(api_key)

    # Prompt template
    prompt_template = create_llm_prompt_template()

    # Prompt data
    prompt_data = prepare_prompt(model_name, dtc_code, engine_parameters, dtc_query_result, manual_sections)

    # LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate report
    report = llm_chain.run(prompt_data)
    
    return report

