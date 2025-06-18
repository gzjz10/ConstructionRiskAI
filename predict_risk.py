# predict_risk.py
import json
import sys
import numpy as np
import pandas as pd # Added pandas
import pickle

# --- Model specific constants (from your new training script) ---
# These must match the features the 'best_logistic_regression_pipeline.pkl' was trained on.
NEW_MODEL_NUMERIC_FEATURES = [
    'structural_complexity_rating', 'temporary_structures_rating', 'bim_adoption_percentage',
    'lead_time_rating', 'tariff_rate_percentage', 'supply_chain_risk_percentage',
    'socio_political_risk_score', 'permit_approval_rating', 'lien_claims_rating',
    'seismic_zone_rating', 'flood_risk_percentage', 'rainfall_flood_risk_percentage',
    'contingency_budget_percentage', 'weather_delay_rating', 'safety_risk_rating',
    'worker_experience_rating', 'turnover_rating', 'cybersecurity_risk_rating',
    'renewable_energy_percentage', 'energy_efficiency_percentage'
]

NEW_MODEL_CATEGORICAL_FEATURES = ['project_type', 'project_size', 'region']

# Mapping from GUI input names to the names used in the training script for the new model
GUI_TO_MODEL_FEATURE_MAP = {
    "bim_adoption_level_percent": "bim_adoption_percentage",
    "tariff_rate_percent": "tariff_rate_percentage",
    "supply_chain_disruption_risk_percent": "supply_chain_risk_percentage",
    "flood_risk_probability_percent": "flood_risk_percentage",
    "rainfall_flood_risk_percent": "rainfall_flood_risk_percentage",
    "contingency_budget_percent": "contingency_budget_percentage",
    "worker_exp_rating": "worker_experience_rating", # Matches one in training list
    "cybersecurity_risk_assessment": "cybersecurity_risk_rating",
    "renewable_energy_contribution_percent": "renewable_energy_percentage",
    "energy_efficiency_compliance_percent": "energy_efficiency_percentage"
    # Add other direct matches if any, or if names are identical, they don't need explicit mapping
}
# Add direct matches for features where GUI name is same as model training name
for feature in NEW_MODEL_NUMERIC_FEATURES + NEW_MODEL_CATEGORICAL_FEATURES:
    if feature not in GUI_TO_MODEL_FEATURE_MAP.values() and feature not in GUI_TO_MODEL_FEATURE_MAP.keys():
         # If the feature name is used directly in the model and not in the map's values yet,
         # and it's not already a key (meaning it's not a GUI name that needs mapping)
         # then assume GUI name = model name
        is_gui_name_already_a_key = any(gui_name == feature for gui_name in GUI_TO_MODEL_FEATURE_MAP.keys())
        if not is_gui_name_already_a_key:
            GUI_TO_MODEL_FEATURE_MAP[feature] = feature


# Default values for categorical features not provided by GUI
DEFAULT_CATEGORICAL_VALUES = {
    "project_type": "General Construction", # Ensure this is a category seen during training or OHE handles unknowns
    "project_size": "Medium",
    "region": "North America"
}

# Load model classes for probability mapping
try:
    with open('model_classes.json', 'r') as f:
        MODEL_CLASSES_ORDER = json.load(f)
except FileNotFoundError:
    print("Error: model_classes.json not found. Please run save_best_lr_model.py first.", file=sys.stderr)
    # Fallback, assuming a common order; this might be incorrect if the model's classes_ is different
    MODEL_CLASSES_ORDER = ['Low', 'Medium', 'High'] # Or based on your actual class names

# --- Rule-based definitions (can remain as they are for supplementary info) ---
RISK_FACTOR_RULES_DEF = [
    {
        "paramName": "seismic_zone_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High seismic zone rating", "mediumRiskDesc": "Moderate seismic zone rating", "lowRiskDesc": "Low seismic zone rating",
        "highRecommendation": {"risk": "High seismic activity in project area", "suggestion": "Engage specialized seismic engineering consultants and implement enhanced structural reinforcement designs.", "priority": 9},
        "mediumRecommendation": {"risk": "Moderate seismic activity in project area", "suggestion": "Implement additional seismic design standards beyond minimum code requirements.", "priority": 6},
        "lowRecommendation": {"risk": "Low seismic risk", "suggestion": "Follow standard building codes and maintain regular seismic safety inspections.", "priority": 3}
    },
    {
        "paramName": "flood_risk_probability_percent", "highThreshold": 25, "mediumThreshold": 10, "checkGreaterThan": True,
        "highRiskDesc": "High flood risk probability", "mediumRiskDesc": "Moderate flood risk probability", "lowRiskDesc": "Low flood risk probability",
        "highRecommendation": {"risk": "High flood risk in project area", "suggestion": "Develop comprehensive flood mitigation plans including elevated foundations, waterproof materials, and emergency water diversion systems.", "priority": 8},
        "mediumRecommendation": {"risk": "Moderate flood risk in project area", "suggestion": "Incorporate flood-resistant design elements and improved site drainage systems.", "priority": 5},
        "lowRecommendation": {"risk": "Low flood risk", "suggestion": "Maintain standard drainage systems and monitor weather forecasts during construction.", "priority": 2}
    },
    {
        "paramName": "structural_complexity_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High structural complexity", "mediumRiskDesc": "Moderate structural complexity", "lowRiskDesc": "Low structural complexity",
        "highRecommendation": {"risk": "Complex structural engineering requirements", "suggestion": "Conduct advanced engineering peer reviews and increase frequency of structural quality control inspections.", "priority": 7},
        "mediumRecommendation": {"risk": "Multiple structural engineering challenges", "suggestion": "Implement specialized structural monitoring during construction phases.", "priority": 5},
        "lowRecommendation": {"risk": "Minimal structural challenges", "suggestion": "Proceed with standard engineering review processes.", "priority": 2}
    },
    {
        "paramName": "bim_adoption_level_percent", "highThreshold": 40, "mediumThreshold": 70, "checkGreaterThan": False, 
        "highRiskDesc": "Low BIM adoption level (<=40%)", "mediumRiskDesc": "Moderate BIM adoption level (41-70%)", "lowRiskDesc": "High BIM adoption level (>70%)",
        "highRecommendation": {"risk": "Insufficient BIM implementation", "suggestion": "Increase BIM adoption to at least 70% and provide training to improve coordination and reduce engineering conflicts.", "priority": 7},
        "mediumRecommendation": {"risk": "Partial BIM implementation", "suggestion": "Increase BIM implementation to at least 80% for critical project components.", "priority": 4},
        "lowRecommendation": {"risk": "Adequate BIM implementation", "suggestion": "Continue current BIM practices and consider expanding to additional project aspects.", "priority": 2}
    },
    {
        "paramName": "lead_time_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "Long lead time for critical materials", "mediumRiskDesc": "Moderate lead time for critical materials", "lowRiskDesc": "Short lead time for critical materials",
        "highRecommendation": {"risk": "Extended lead times for critical materials", "suggestion": "Identify alternative suppliers and consider local sourcing options to reduce lead times. Implement early procurement protocols.", "priority": 8},
        "mediumRecommendation": {"risk": "Moderate lead times for materials", "suggestion": "Initiate early procurement for critical materials and establish supplier performance tracking.", "priority": 5},
        "lowRecommendation": {"risk": "Short lead times", "suggestion": "Maintain current procurement schedule with standard monitoring procedures.", "priority": 2}
    },
    {
        "paramName": "supply_chain_disruption_risk_percent", "highThreshold": 30, "mediumThreshold": 15, "checkGreaterThan": True,
        "highRiskDesc": "High supply chain disruption risk", "mediumRiskDesc": "Moderate supply chain disruption risk", "lowRiskDesc": "Low supply chain disruption risk",
        "highRecommendation": {"risk": "High likelihood of supply chain disruptions", "suggestion": "Implement dual-sourcing strategy for critical materials and establish material stockpiling protocol.", "priority": 9},
        "mediumRecommendation": {"risk": "Potential supply chain disruptions", "suggestion": "Develop supply chain disruption contingency plans and monitor supplier financial stability.", "priority": 6},
        "lowRecommendation": {"risk": "Low supply chain risk", "suggestion": "Continue standard procurement processes with routine supplier evaluations.", "priority": 2}
    },
    {
        "paramName": "socio_political_risk_score", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High socio-political risk", "mediumRiskDesc": "Moderate socio-political risk", "lowRiskDesc": "Low socio-political risk",
        "highRecommendation": {"risk": "Significant political instability and community opposition", "suggestion": "Establish comprehensive stakeholder management plan and engage a community liaison officer.", "priority": 8},
        "mediumRecommendation": {"risk": "Moderate political and community concerns", "suggestion": "Increase community engagement activities and develop political risk mitigation strategies.", "priority": 5},
        "lowRecommendation": {"risk": "Stable socio-political environment", "suggestion": "Maintain standard community relations and monitoring of political developments.", "priority": 2}
    },
    {
        "paramName": "contingency_budget_percent", "highThreshold": 10, "mediumThreshold": 15, "checkGreaterThan": False, 
        "highRiskDesc": "Low contingency budget (<=10%)", "mediumRiskDesc": "Moderate contingency budget (11-15%)", "lowRiskDesc": "Adequate contingency budget (>15%)",
        "highRecommendation": {"risk": "Insufficient contingency budget", "suggestion": "Increase contingency budget to minimum 15% to account for identified risk factors.", "priority": 9},
        "mediumRecommendation": {"risk": "Limited contingency budget", "suggestion": "Consider raising contingency to 15-20% and implement enhanced cost control measures.", "priority": 6},
        "lowRecommendation": {"risk": "Sufficient contingency budget", "suggestion": "Maintain current contingency allocation with standard budget monitoring.", "priority": 3}
    },
    {
        "paramName": "safety_risk_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High safety risk rating", "mediumRiskDesc": "Moderate safety risk rating", "lowRiskDesc": "Low safety risk rating",
        "highRecommendation": {"risk": "Poor safety record with multiple incidents", "suggestion": "Implement comprehensive safety audit program and dedicated safety training initiatives.", "priority": 9},
        "mediumRecommendation": {"risk": "Several safety incidents in past year", "suggestion": "Enhance safety monitoring protocols and increase frequency of safety inspections.", "priority": 7},
        "lowRecommendation": {"risk": "Good safety record", "suggestion": "Continue current safety programs with regular refresher training.", "priority": 3}
    },
    {
        "paramName": "permit_approval_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "Slow permit approval", "mediumRiskDesc": "Moderate permit approval time", "lowRiskDesc": "Fast permit approval",
        "highRecommendation": {"risk": "Slow permit approval expected", "suggestion": "Engage with authorities early and submit applications well in advance", "priority": 7},
        "mediumRecommendation": {"risk": "Moderate permit approval time", "suggestion": "Submit applications with buffer time", "priority": 4},
        "lowRecommendation": {"risk": "Fast permit approval", "suggestion": "Proceed with standard application process", "priority": 2}
    },
    {
        "paramName": "lien_claims_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High lien claims risk", "mediumRiskDesc": "Moderate lien claims risk", "lowRiskDesc": "Low lien claims risk",
        "highRecommendation": {"risk": "History of multiple lien claims", "suggestion": "Implement enhanced contract management and payment tracking systems", "priority": 6},
        "mediumRecommendation": {"risk": "Some past lien claims", "suggestion": "Increase oversight of subcontractor payments", "priority": 4},
        "lowRecommendation": {"risk": "No significant lien history", "suggestion": "Maintain standard payment processes", "priority": 2}
    },
    {
        "paramName": "weather_delay_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High weather delay risk", "mediumRiskDesc": "Moderate weather delay risk", "lowRiskDesc": "Low weather delay risk",
        "highRecommendation": {"risk": "High likelihood of weather delays", "suggestion": "Increase schedule buffers and implement weather monitoring system", "priority": 7},
        "mediumRecommendation": {"risk": "Potential weather delays", "suggestion": "Add moderate schedule buffers", "priority": 4},
        "lowRecommendation": {"risk": "Minimal weather risk", "suggestion": "Standard scheduling practices sufficient", "priority": 2}
    },
    { 
        "paramName": "worker_exp_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True, 
        "highRiskDesc": "Inexperienced workforce (rating 4-5)", "mediumRiskDesc": "Moderately experienced workforce (rating 3)", "lowRiskDesc": "Experienced workforce (rating 1-2)",
        "highRecommendation": {"risk": "Workforce lacks experience", "suggestion": "Implement enhanced training programs and increase supervision", "priority": 8},
        "mediumRecommendation": {"risk": "Mixed experience levels", "suggestion": "Provide targeted training where needed", "priority": 5},
        "lowRecommendation": {"risk": "Experienced workforce", "suggestion": "Maintain current training programs", "priority": 2}
    },
    {
        "paramName": "turnover_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High turnover risk", "mediumRiskDesc": "Moderate turnover risk", "lowRiskDesc": "Low turnover risk",
        "highRecommendation": {"risk": "High worker turnover expected", "suggestion": "Implement retention programs and improve working conditions", "priority": 7},
        "mediumRecommendation": {"risk": "Moderate turnover expected", "suggestion": "Monitor workforce satisfaction", "priority": 4},
        "lowRecommendation": {"risk": "Stable workforce", "suggestion": "Continue current workforce policies", "priority": 2}
    },
    {
        "paramName": "tariff_rate_percent", "highThreshold": 15, "mediumThreshold": 8, "checkGreaterThan": True,
        "highRiskDesc": "High tariff rates", "mediumRiskDesc": "Moderate tariff rates", "lowRiskDesc": "Low tariff rates",
        "highRecommendation": {"risk": "High tariff rates on materials", "suggestion": "Explore alternative suppliers or local sourcing options to mitigate tariff impacts.", "priority": 6},
        "mediumRecommendation": {"risk": "Moderate tariff rates", "suggestion": "Monitor tariff trends and consider pre-purchasing key materials.", "priority": 4},
        "lowRecommendation": {"risk": "Low tariff impact", "suggestion": "Maintain current procurement strategy with tariff monitoring.", "priority": 2}
    },
    {
        "paramName": "cybersecurity_risk_assessment", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True, # GUI uses 'assessment', model used 'rating'
        "highRiskDesc": "High cybersecurity risk", "mediumRiskDesc": "Moderate cybersecurity risk", "lowRiskDesc": "Low cybersecurity risk",
        "highRecommendation": {"risk": "Significant cybersecurity vulnerabilities", "suggestion": "Implement enhanced cybersecurity measures including network segmentation, multi-factor authentication, and regular penetration testing.", "priority": 7},
        "mediumRecommendation": {"risk": "Some cybersecurity vulnerabilities", "suggestion": "Conduct security audit and implement basic protections like firewalls and employee training.", "priority": 5},
        "lowRecommendation": {"risk": "Strong cybersecurity posture", "suggestion": "Maintain current security protocols with regular reviews.", "priority": 2}
    },
    {
        "paramName": "renewable_energy_contribution_percent", "highThreshold": 30, "mediumThreshold": 50, "checkGreaterThan": False, 
        "highRiskDesc": "Low renewable energy contribution (<=30%)", "mediumRiskDesc": "Moderate renewable energy contribution (31-50%)", "lowRiskDesc": "High renewable energy contribution (>50%)",
        "highRecommendation": {"risk": "Insufficient renewable energy usage", "suggestion": "Explore additional renewable energy options to meet sustainability goals and potential regulations.", "priority": 5},
        "mediumRecommendation": {"risk": "Moderate renewable energy usage", "suggestion": "Consider incremental increases in renewable energy sources.", "priority": 3},
        "lowRecommendation": {"risk": "Strong renewable energy usage", "suggestion": "Continue current renewable energy strategy.", "priority": 1}
    },
    {
        "paramName": "energy_efficiency_compliance_percent", "highThreshold": 70, "mediumThreshold": 85, "checkGreaterThan": False, 
        "highRiskDesc": "Low energy efficiency compliance (<=70%)", "mediumRiskDesc": "Moderate energy efficiency compliance (71-85%)", "lowRiskDesc": "High energy efficiency compliance (>85%)",
        "highRecommendation": {"risk": "Energy efficiency below standards", "suggestion": "Conduct energy audit and implement efficiency improvements to meet compliance requirements.", "priority": 6},
        "mediumRecommendation": {"risk": "Partial energy efficiency compliance", "suggestion": "Identify and address key areas needing improvement.", "priority": 4},
        "lowRecommendation": {"risk": "Full energy efficiency compliance", "suggestion": "Maintain current energy efficiency measures.", "priority": 2}
    },
    {
        "paramName": "temporary_structures_rating", "highThreshold": 4, "mediumThreshold": 3, "checkGreaterThan": True,
        "highRiskDesc": "High temporary structures requirement", "mediumRiskDesc": "Moderate temporary structures requirement", "lowRiskDesc": "Low temporary structures requirement",
        "highRecommendation": {"risk": "Extensive temporary structures needed", "suggestion": "Develop detailed temporary structures plan with safety and stability assessments.", "priority": 5},
        "mediumRecommendation": {"risk": "Several temporary structures needed", "suggestion": "Plan for adequate temporary facilities with standard safety measures.", "priority": 3},
        "lowRecommendation": {"risk": "Minimal temporary structures needed", "suggestion": "Standard temporary structures planning sufficient.", "priority": 2}
    },
    {
        "paramName": "rainfall_flood_risk_percent", "highThreshold": 25, "mediumThreshold": 10, "checkGreaterThan": True,
        "highRiskDesc": "High rainfall flood risk", "mediumRiskDesc": "Moderate rainfall flood risk", "lowRiskDesc": "Low rainfall flood risk",
        "highRecommendation": {"risk": "High risk of rainfall flooding", "suggestion": "Implement comprehensive drainage systems and flood barriers for vulnerable areas.", "priority": 7},
        "mediumRecommendation": {"risk": "Moderate risk of rainfall flooding", "suggestion": "Improve site drainage and prepare emergency response plans.", "priority": 4},
        "lowRecommendation": {"risk": "Low risk of rainfall flooding", "suggestion": "Maintain standard drainage monitoring procedures.", "priority": 2}
    }
]
# No need for MODEL_FEATURE_ORDER from the old model now. The new model pipeline handles features.


def prepare_input_for_new_model(gui_risk_data_dict):
    """
    Transforms the flat dictionary from the GUI into a Pandas DataFrame
    structured as expected by the new Logistic Regression model's preprocessor.
    """
    model_input_data = {}

    # Map numeric features
    for gui_key, model_key in GUI_TO_MODEL_FEATURE_MAP.items():
        if model_key in NEW_MODEL_NUMERIC_FEATURES:
            model_input_data[model_key] = gui_risk_data_dict.get(gui_key, 0) # Default to 0 if missing

    # Add default categorical features
    for cat_feature in NEW_MODEL_CATEGORICAL_FEATURES:
        model_input_data[cat_feature] = DEFAULT_CATEGORICAL_VALUES.get(cat_feature, "Unknown")

    # Create a single-row DataFrame
    # The order of columns in this DataFrame must match what the preprocessor was fit on.
    # The preprocessor (ColumnTransformer) uses the lists NEW_MODEL_NUMERIC_FEATURES and NEW_MODEL_CATEGORICAL_FEATURES
    # to select and order columns.
    
    # Create an ordered list of all features the preprocessor expects
    all_expected_features = NEW_MODEL_NUMERIC_FEATURES + NEW_MODEL_CATEGORICAL_FEATURES
    
    # Create a dictionary for the DataFrame row in the correct order
    df_row = {feature: model_input_data.get(feature) for feature in all_expected_features}

    return pd.DataFrame([df_row], columns=all_expected_features)


def classify_risk_factors(risk_data_dict): # This function remains useful for supplementary info
    risk_infos = []
    for rule_def in RISK_FACTOR_RULES_DEF:
        param_name = rule_def["paramName"]
        if param_name not in risk_data_dict:
            continue
        
        value = risk_data_dict[param_name]
        is_high = (value >= rule_def["highThreshold"] if rule_def["checkGreaterThan"]
                   else value <= rule_def["highThreshold"])
        is_medium = ((value >= rule_def["mediumThreshold"] and value < rule_def["highThreshold"]) if rule_def["checkGreaterThan"]
                     else (value <= rule_def["mediumThreshold"] and value > rule_def["highThreshold"]))

        if is_high:
            risk_infos.append({
                "category": "high", "description": rule_def["highRiskDesc"],
                "recommendation": rule_def["highRecommendation"]
            })
        elif is_medium:
            risk_infos.append({
                "category": "medium", "description": rule_def["mediumRiskDesc"],
                "recommendation": rule_def["mediumRecommendation"]
            })
        else:
            risk_infos.append({
                "category": "low", "description": rule_def["lowRiskDesc"],
                "recommendation": rule_def["lowRecommendation"]
            })
            
    low_count = sum(1 for r in risk_infos if r["category"] == "low")
    medium_count = sum(1 for r in risk_infos if r["category"] == "medium")
    high_count = sum(1 for r in risk_infos if r["category"] == "high")
    
    return low_count, medium_count, high_count, risk_infos

# The EnhancedEffectiveScoreCalculator and related functions can be kept for supplementary info
# but they are NOT used for the primary prediction.
# For brevity, I'll omit them here, but you can paste them back if you want that score calculated.
# For now, let's return a dummy effective_score.
def calculate_dummy_effective_score():
    return 0.0

def predict_risk_combined(gui_risk_data_dict):
    # Prepare input for the new Logistic Regression model
    model_input_df = prepare_input_for_new_model(gui_risk_data_dict.copy())
    
    predicted_risk_class_str = "Medium" # Default
    model_probabilities_ordered_by_class_ = [0.2, 0.6, 0.2] # Default P(Low), P(Medium), P(High)

    try:
        with open('best_logistic_regression_pipeline.pkl', 'rb') as f:
            loaded_model_pipeline = pickle.load(f)
        
        # Predict class and probabilities
        predicted_risk_class_str = loaded_model_pipeline.predict(model_input_df)[0]
        proba_from_model = loaded_model_pipeline.predict_proba(model_input_df)[0]

        # Map probabilities to fixed order: Low, Medium, High
        # MODEL_CLASSES_ORDER was loaded from model_classes.json
        # GUI expects probabilities for [P(Low), P(Medium), P(High)]
        # Let's assume your model's classes are typically 'Low', 'Medium', 'High' or similar
        # and the target risk categories in your problem are 0 (Low), 1 (Medium), 2 (High)

        # Standardized internal class names for mapping
        app_risk_labels = ['Low', 'Medium', 'High'] # Target order for GUI
        model_probabilities_ordered_by_class_ = [0.0] * len(app_risk_labels)

        for i, class_name_from_model in enumerate(MODEL_CLASSES_ORDER): # MODEL_CLASSES_ORDER is from model_classes.json
            if class_name_from_model in app_risk_labels:
                idx_in_app_order = app_risk_labels.index(class_name_from_model)
                model_probabilities_ordered_by_class_[idx_in_app_order] = proba_from_model[i]
            else:
                # Handle case where model class name doesn't match expected app labels
                # This shouldn't happen if model_classes.json is correct
                print(f"Warning: Model class '{class_name_from_model}' not in standard app labels {app_risk_labels}", file=sys.stderr)


    except Exception as e:
        sys.stderr.write(f"Error loading or using the new Logistic Regression model: {e}\\nUsing fallback assessment.\\n")
        # Basic fallback
        if gui_risk_data_dict.get("safety_risk_rating", 0) >= 4 or gui_risk_data_dict.get("seismic_zone_rating",0) >=4 :
            predicted_risk_class_str = "High"
            model_probabilities_ordered_by_class_ = [0.1, 0.2, 0.7]
        elif gui_risk_data_dict.get("supply_chain_disruption_risk_percent", 0) > 25:
            predicted_risk_class_str = "Medium"
            model_probabilities_ordered_by_class_ = [0.2, 0.6, 0.2]
        else:
            predicted_risk_class_str = "Low"
            model_probabilities_ordered_by_class_ = [0.7, 0.2, 0.1]


    # Map string prediction to numeric for the GUI
    # Assuming your Logistic Regression model predicts string labels like "Low", "Medium", "High"
    # This mapping must align with how your y_true was encoded if it was numeric
    prediction_numeric = 1 # Default to medium
    if predicted_risk_class_str == 'Low': # Or your model's actual label for low
        prediction_numeric = 0
    elif predicted_risk_class_str == 'High': # Or your model's actual label for high
        prediction_numeric = 2
    elif predicted_risk_class_str == 'Medium': # Or your model's actual label for medium
        prediction_numeric = 1
    # Add more specific mappings if your model outputs different string labels

    # Rule-based classification for supplementary information
    low_count, medium_count, high_count, risk_infos_detail = classify_risk_factors(gui_risk_data_dict)
    
    effective_score = calculate_dummy_effective_score() # Placeholder

    return int(prediction_numeric), model_probabilities_ordered_by_class_, risk_infos_detail, low_count, medium_count, high_count, effective_score


def main():
    input_json = sys.stdin.read()
    gui_risk_data_input = json.loads(input_json) # Renamed for clarity
    
    prediction, probabilities, risk_details, lc, mc, hc, eff_score = predict_risk_combined(gui_risk_data_input)
    
    result = {
        "prediction": prediction,
        "probabilities": probabilities,
        "risk_infos": risk_details,
        "low_count": lc,
        "medium_count": mc,
        "high_count": hc,
        "effective_score": eff_score
    }
    sys.stdout.write(json.dumps(result))

if __name__ == "__main__":
    main()
