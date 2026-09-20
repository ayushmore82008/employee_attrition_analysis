import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pydeck as pdk
from faker import Faker  # For generating fake avatars
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score
import os
import joblib  

# Define paths relative to the script location
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_best.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
APP_VERSION = "v0.2"
COLOR_THEME = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "background": "#F8F9FA",
    "success": "#6BCB77",
    "warning": "#FFD93D",
    "danger": "#FF6B6B"
}

# Initialize session state for file upload and data
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "true_labels" not in st.session_state:
    st.session_state.true_labels = None


# --- Model Loading ---
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)
@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as scaler_file:
        return pickle.load(scaler_file)
model = load_model()
scaler = load_scaler()
fake = Faker()

# --- UI Configuration ---
def apply_custom_theme():
    st.markdown(f"""
    <style>
    :root {{
        --primary: {COLOR_THEME['primary']};
        --secondary: {COLOR_THEME['secondary']};
        --background: {COLOR_THEME['background']};
    }}
    .stDataFrame {{
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .stButton>button {{
        background-color: var(--primary);
        color: white;
        border-radius: 5px;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: var(--secondary);
        transform: scale(1.05);
    }}
    .metric-card {{
        padding: 1.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# --- Navigation & Progress Tracking ---
st.sidebar.title("🚀 Navigation")
sections = [
        "🏠 Overview", 
        "📤 Upload Data", 
        "📊 Model Evaluation", 
        "📈 Feature Importance",
        "📋 Recommendations",
        "🏢 Work Model - Remote VS Office"
    ]
    
current_section = st.sidebar.radio("Go to", sections)
    
    # Progress tracking
progress_data = {
        "🏠 Overview": (1, "Introduction"),
        "📤 Upload Data": (2, "Data Upload"),
        "📊 Model Evaluation": (3, "Model Analysis"),
        "📈 Feature Importance": (4, "Feature Insights"),
        "📋 Recommendations": (5, "Action Plan"),
        "🏢 Work Model - Remote VS Office": (6, "Work Model Analysis")
    }
    
current_step, step_label = progress_data.get(current_section, (1, ""))
st.sidebar.markdown(f"""
    <div class="metric-card">
        <h4>Progress Tracking</h4>
        <p>Step {current_step}/6 - {step_label}</p>
        <progress value="{current_step}" max="6"></progress>
    </div>
    """, unsafe_allow_html=True)
    
st.sidebar.markdown("---")
st.sidebar.markdown(f"ℹ️ App Version: {APP_VERSION}")
st.sidebar.markdown("💻 [Project GitHub](https://github.com/J0hnrusso)")
st.sidebar.markdown("🔗 [Connect on LinkedIn](https://www.linkedin.com/in/joaorussofigueiredo/)")
    
# Functions
def display_overview():
    st.title("Employee Attrition Prediction App")
    st.markdown("## 🏠 Overview")
    st.write("""
    This application helps predict employee attrition risk and provides actionable insights. Employee turnover (also known as "employee churn") is a costly problem for companies. The true cost of replacing an employee
    can often be quite large. This is due to the amount of time spent to interview and find a replacement, sign-on bonuses, and the loss of productivity for several months while the new employee gets accustomed to the new role.
    """)
    
    # Step-by-step guide
    with st.expander("📖 Getting Started Guide", expanded=True):
        steps = """
        1. **Upload Data**: Provide your employee dataset in CSV format
        2. **Model Evaluation**: Review model performance metrics
        3. **Feature Analysis**: Understand key attrition drivers
        4. **Recommendations**: Get personalized action plans
        """
        st.markdown(steps)
    
    # Quick stats if data exists
    if st.session_state.uploaded_data is not None:
        st.markdown("### 🚨 Current Dataset Snapshot")
        data = st.session_state.uploaded_data
        risk_counts = data['Risk Category'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Employees Analyzed</h4>
                <h2>{len(data)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>High-Risk Employees</h4>
                <h2>{risk_counts.get('High-risk', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Attrition Risk Score</h4>
                <h2>{st.session_state.predictions.mean() * 100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

# --- Upload Data Section ---
def display_upload_data():
    st.markdown("## 📤 Data Upload & Setup")
    
    # Display required features and data types
    with st.expander("📋 Data Requirements", expanded=True):
        required_features = get_required_features_from_model(model)
        feature_data_types = {
                            'Age': 'int64',
                            'DailyRate': 'int64',
                            'DistanceFromHome': 'int64',
                            'Education': 'int64',
                            'EmployeeNumber': 'int64',
                            'EnvironmentSatisfaction': 'int64',
                            'JobInvolvement': 'int64',
                            'JobLevel': 'int64',
                            'JobSatisfaction': 'int64',
                            'MonthlyIncome': 'int64',
                            'OverTime': 'bool',
                            'StockOptionLevel': 'int64',
                            'TotalWorkingYears': 'int64',
                            'TrainingTimesLastYear': 'int64',
                            'WorkLifeBalance': 'int64',
                            'YearsAtCompany': 'int64',
                            'YearsInCurrentRole': 'int64',
                            'YearsWithCurrManager': 'int64',
                            'BusinessTravel_Travel_Frequently': 'bool',
                            'BusinessTravel_Travel_Rarely': 'bool',
                            'MaritalStatus_Married': 'bool',
                            'MaritalStatus_Single': 'bool'
        }

        st.table(pd.DataFrame({
            "Feature": required_features,
            "Data Type": [feature_data_types[feature] for feature in required_features]
        }))

        # Download template
        if st.button("✨ Generate Custom Template"):
            custom_template = pd.DataFrame(columns=required_features)
            csv_template = custom_template.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Template",
                data=csv_template,
                file_name="data_template.csv",
                mime="text/csv"
            )

    # Enhanced upload zone
    with st.container():
        uploaded_file = st.file_uploader(
            "Drag CSV here or click to browse",
            type="csv",
            help="Ensure file matches required features below"
        )
        
        # Ensure file is uploaded before proceeding
        if uploaded_file is not None:
            try:
                # Read and store uploaded file in session state
                data = pd.read_csv(uploaded_file)
                
                if data.empty:
                    st.error("Uploaded CSV file is empty. Please upload a valid file.")
                    return
                
                # Store the file in session state (persists across pages)
                st.session_state.uploaded_data = data
                st.success("File successfully uploaded and stored!")

                # Display preview in an expander
                with st.expander("🔍 Data Preview", expanded=True):
                    st.dataframe(data.head(5), use_container_width=True)

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return

        # Form to process the uploaded data
        if "uploaded_data" in st.session_state:
            with st.form("data_processing_form"):
                st.markdown("### ⚙️ Data Processing Options")
                
                use_scaler = st.checkbox("Apply feature scaling - StandardScaler", value=False)
                have_true_labels = st.checkbox("File contains True labels", value=False)
                
                if st.form_submit_button("Process Data"):
                    with st.spinner("Analyzing data..."):
                        process_and_store_data(st.session_state.uploaded_data, use_scaler, have_true_labels)
                    st.success("Data processed successfully!")

def process_and_store_data(data, use_scaler, have_true_labels):
    """
    Processes uploaded data, checks required features, applies scaling (if selected),
    and stores predictions & risk categories in session state.
    """
    try:
        required_features = get_required_features_from_model(model)  # Ensure this function returns a list of features

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Uploaded file could not be read as a DataFrame.")

        # Check for missing features
        missing_features = [f for f in required_features if f not in list(data.columns)]
        if missing_features:
            raise ValueError(f"Missing features: {', '.join(missing_features)}")

        # Extract true labels if present
        true_labels = data["Attrition"] if have_true_labels and "Attrition" in data.columns else None
        if have_true_labels and "Attrition" in data.columns:
            data = data.drop(columns=["Attrition"])

        # Prepare data for prediction
        data_for_prediction = data[required_features]
        if use_scaler:
            data_scaled = scaler.transform(data_for_prediction)
        else:
            data_scaled = data_for_prediction

        # Make predictions
        predictions = model.predict_proba(data_scaled)[:, 1]

        # Categorize risk levels
        data['Risk Category'] = np.select(
            [predictions < 0.5, (predictions >= 0.5) & (predictions <= 0.75), predictions > 0.75],
            ["Low-risk", "Medium-risk", "High-risk"]
        )

        # Store results in session state
        st.session_state.uploaded_data = data
        st.session_state.predictions = predictions
        st.session_state.true_labels = true_labels
        st.session_state.processed = True  # Mark data as processed

        st.success("✅ Data processed and stored successfully!")

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")


# --- Helper Function ---
def get_required_features_from_model(model):
    """Dynamically extract required features from the trained model."""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    else:
        # Fallback to a predefined list if the model doesn't expose feature names
        return ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',
       'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'OverTime', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager',
       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'MaritalStatus_Married', 'MaritalStatus_Single'
        ]

# Display model evaluation section
def display_model_evaluation():
    st.markdown("## 📊 Model Performance Analysis")
    
    if st.session_state.uploaded_data is not None:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("### Key Metrics")
            if st.session_state.true_labels is not None:
                y_true = st.session_state.true_labels
                y_pred = (st.session_state.predictions > 0.5).astype(int)
                st.write("Accuracy:", round(accuracy_score(y_true, y_pred), 2))
                st.write("Precision:", round(average_precision_score(y_true, y_pred), 2))
                st.write("Recall:", round(recall_score(y_true, y_pred), 2))
            else:
                st.warning("No true labels available for metrics calculation")

        with col2:
            if st.session_state.true_labels is not None:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Stay", "Leave"],
                            yticklabels=["Stay", "Leave"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

def display_recommendations():
    st.markdown("## 📋 Actionable Insights")
    
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        risk_counts = data['Risk Category'].value_counts()
        retired_employees = data[data['Age'] >= 60].shape[0]

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### Risk Distribution")
            fig = px.sunburst(
                data,
                path=['Risk Category'],
                color='Risk Category',
                color_discrete_map={
                    'High-risk': '#FF6B6B',
                    'Medium-risk': '#FFD93D',
                    'Low-risk': '#6BCB77'
                },
                hover_data=['MonthlyIncome'],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Priority Actions")
            st.write(f"🚨 High-risk employees: {risk_counts.get('High-risk', 0)}")
            st.write(f"⚠️ Medium-risk employees: {risk_counts.get('Medium-risk', 0)}")
            st.write(f"✅ Low-risk employees: {risk_counts.get('Low-risk', 0)}")
            st.write(f"🔜 Close to Retirement: {retired_employees}")
        st.markdown("### High-Risk Employee - Action Plans:")
        high_risk = data[data['Risk Category'] == 'High-risk']
        
        if not high_risk.empty:
            with st.expander("Filter High-Risk Employees", expanded=True):
                min_salary = st.slider(
                    "Monthly Income $", 
                    int(data['MonthlyIncome'].min()),
                    int(data['MonthlyIncome'].max()),
                    int(data['MonthlyIncome'].median())
                )
                filtered_data = high_risk[high_risk['MonthlyIncome'] >= min_salary]

            for EmployeeNumber, row in filtered_data.iterrows():
                with st.expander(f"Employee {EmployeeNumber} Profile", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Distance To Work:** ~{row.get('DistanceFromHome', 'N/A')} Km")
                        st.markdown(f"**Has Stock Options:** {row.get('StockOptionLevel', 'N/A')}")
                        st.markdown(f"**Tenure:** {row['YearsAtCompany']} years")
                    with col2:
                        st.markdown(f"**Self reported Satisfaction:** {row['JobSatisfaction']}/4")
                        st.markdown(f"**Income:** ${row['MonthlyIncome']:,.0f}")
                        st.markdown(f"**Working Overtime?** {row.get('OverTime', 'N/A')}")
                    st.markdown("#### Action Items")
                    actions = []
                    if row['JobSatisfaction'] < 3:
                        actions.append("Conduct a one-on-one retention interview to understand concerns and improve job satisfaction")
                    if row['YearsAtCompany'] < 2:
                        actions.append("Assign to a mentorship program to support onboarding and career development")
                    if row['MonthlyIncome'] < data['MonthlyIncome'].median():
                        actions.append("Review and adjust compensation package to ensure competitiveness and fairness")
                    if row['StockOptionLevel'] < data['StockOptionLevel'].median():
                        actions.append("Evaluate and adjust stock option levels to enhance employee incentives")
                    if row['DistanceFromHome'] > data['DistanceFromHome'].median():
                        actions.append("Explore remote work options or provide transportation support to reduce commute stress")
                    if row['OverTime'] > 0:
                        actions.append("Review workload and responsibilities to ensure a healthy work-life balance and reduce overtime")
                    if actions:
                        for action in actions:
                            st.markdown(f"- 🎯 {action}")
                    else:
                        st.info("No specific actions recommended - monitor regularly")
        else:
            st.success("🎉 No high-risk employees detected!")
    
    st.markdown("### Employees Close to Retirement 🔜")
    retired_employees = data[data['Age'] >= 60]

    if not retired_employees.empty:
            with st.expander("Filter Employees Close to Retirement", expanded=True):
                min_age = 60  # Filtering employees who are older than 60
                filtered_data_retired = retired_employees[retired_employees['Age'] >= min_age]

            for EmployeeNumber, row in filtered_data_retired.iterrows():
                with st.expander(f"Employee {EmployeeNumber} Profile", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Age:** {row['Age']} years")
                        st.markdown(f"**Tenure:** {row['YearsAtCompany']} years")
                        st.markdown(f"**Distance To Work:** ~{row.get('DistanceFromHome', 'N/A')} Km")
                    with col2:
                        st.markdown(f"**Job Satisfaction:** {row['JobSatisfaction']}/4")
                        st.markdown(f"**Income:** ${row['MonthlyIncome']:,.0f}")
                    
                    st.markdown("#### Action Items for Retirement Planning")
                    actions = []
                    if row['Age'] > 60:
                        actions.append("Start retirement planning discussions")
                    if row['YearsAtCompany'] < 5:
                        actions.append("Encourage knowledge transfer and succession planning")
                    if row['JobSatisfaction'] < 3:
                        actions.append("Offer retirement perks and consult on career satisfaction")
                    if row['DistanceFromHome'] > data['DistanceFromHome'].median():
                        actions.append("Consider remote work options or transportation support")

                    if actions:
                        for action in actions:
                            st.markdown(f"- 🎯 {action}")
                    else:
                        st.info("No specific actions recommended - monitor regularly")
    
# Display feature importance section
def display_feature_importance():
    st.markdown("## 📈 Feature Impact Analysis")
    
    if st.session_state.uploaded_data is not None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = st.session_state.uploaded_data.columns[indices]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top 10 Drivers")
            for i, (feat, imp) in enumerate(zip(features[:10], importances[indices][:10])):
                st.markdown(f"{i+1}. **{feat}** ({imp:.3f})")

        with col2:
            st.markdown("### Feature Importance Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importances[indices][:10], y=features[:10], ax=ax)
            ax.set_title("Top 10 Most Important Features")
            st.pyplot(fig)

def display_work_model_analysis():
    st.markdown("## 🏢 Work Model - Remote VS Office")
    
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        
        # Calculate commute time in minutes (assuming distance in kilometers and average speed in km/h)
        if 'CommuteTime' not in data.columns:
            data['CommuteTime'] = (data['DistanceFromHome'] * 60) / 30  # Commute time in minutes assuming 30 km/h speed
        
        # Transportation mode selection
        transportation_options = ['Bus', 'Metro', 'Car', 'Bike']
        transport_mode = st.selectbox("Select Mode of Transportation", transportation_options)

        # Average commute times based on mode of transportation (in minutes per 1 km)
        transport_commute_times = {
            'Bus': 5,    # 5 minutes per km
            'Metro': 4,  # 4 minutes per km
            'Car': 3,    # 3 minutes per km
            'Bike': 2    # 2 minutes per km
        }

        # Apply the selected mode of transportation to calculate commute times
        data['CommuteTime'] = data['DistanceFromHome'] * transport_commute_times[transport_mode]

        # Distance vs Commute Time Analysis
        with st.expander("Distance From Home vs Commute Time", expanded=True):
            st.markdown("### Distance vs Commute Time for Selected Transportation Mode")
            fig = px.scatter(
                data, 
                x='DistanceFromHome', 
                y='CommuteTime', 
                title=f"Distance From Home vs Commute Time ({transport_mode})",
                labels={"": "Distance From Home (KM)", "CommuteTime": "Commute Time (minutes)"},
                color='CommuteTime',  # Assuming 'Work Model' column exists or can be derived
                color_discrete_map={"Remote": "#FFC107", "Office": "#4CAF50", "Hybrid": "#2196F3"}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Commute Time Distribution
        with st.expander("Commute Time Distribution", expanded=True):
            st.markdown("### Distribution of Employees by Commute Time (in minutes)")
            fig_dist = px.histogram(
                data, 
                x='CommuteTime', 
                nbins=20, 
                title="Distribution of Employees by Commute Time",
                labels={"CommuteTime": "Commute Time (minutes)"}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Employee Work Schedule Recommendations
        with st.expander("Employee Work Schedule Recommendations", expanded=True):
            st.markdown("### Recommended Work Schedule Based on Commute Time")
            
            # Define work model recommendation based on commute time
            data['Recommended Work Model'] = data['CommuteTime'].apply(
                lambda x: 'Remote' if x >= 30 else ('Office' if x < 15 else 'Hybrid')
            )

            # Display schedule data
            schedule_data = data[['EmployeeNumber', 'DistanceFromHome', 'CommuteTime', 'Recommended Work Model']]
            st.dataframe(schedule_data)

            # Insights and Recommendations
            st.markdown("### Key Insights 🔎")
            remote_count = len(data[data['Recommended Work Model'] == "Remote"])
            office_count = len(data[data['Recommended Work Model'] == "Office"])
            hybrid_count = len(data[data['Recommended Work Model'] == "Hybrid"])

            st.write(f"- **{remote_count} employees** are recommended for remote work (Commute Time ≥ 30 minutes).")
            st.write(f"- **{office_count} employees** are recommended for office work (Commute Time < 15 minutes).")
            st.write(f"- **{hybrid_count} employees** have a hybrid work model.")

            st.markdown("### Recommendations 🎯")
            st.write("- Consider remote work for employees with long commutes (over 30 minutes).")
            st.write("- Provide transportation support for employees commuting to the office.")
            st.write("- Implement a hybrid model for employees with moderate commutes (15-30 minutes).")

    else:
        st.warning("Please upload data to analyze work models.")



# Main App Flow
if current_section == "🏠 Overview":
    display_overview()
elif current_section == "📤 Upload Data":
    display_upload_data()
elif current_section == "📊 Model Evaluation":
    display_model_evaluation()
elif current_section == "📈 Feature Importance":
    display_feature_importance()
elif current_section == "📋 Recommendations":
    display_recommendations()
elif current_section == "🏢 Work Model - Remote VS Office":
    display_work_model_analysis()
