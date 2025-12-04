import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

st.set_page_config(
    page_title="Fitness Activity Classifier",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    
    try:
        model = joblib.load('fitness_classifier_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("  Model files not found! Please train the model first using the Jupyter notebook.")
        return None, None

model, scaler = load_model()



def calculate_features(df):
   
    
    if not all(col in df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
        st.error("Data must contain accel_x, accel_y, and accel_z columns")
        return None
    
    
    df['magnitude'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    df['xy_magnitude'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2)
    
    
    window_size = 10
    df['x_rolling_mean'] = df['accel_x'].rolling(window=window_size, min_periods=1).mean()
    df['y_rolling_mean'] = df['accel_y'].rolling(window=window_size, min_periods=1).mean()
    df['z_rolling_mean'] = df['accel_z'].rolling(window=window_size, min_periods=1).mean()
    df['x_rolling_std'] = df['accel_x'].rolling(window=window_size, min_periods=1).std().fillna(0)
    df['y_rolling_std'] = df['accel_y'].rolling(window=window_size, min_periods=1).std().fillna(0)
    
    return df

def predict_activity(X, model, scaler):
    
    feature_columns = ['accel_x', 'accel_y', 'accel_z', 'magnitude', 'xy_magnitude',
                       'x_rolling_mean', 'y_rolling_mean', 'z_rolling_mean',
                       'x_rolling_std', 'y_rolling_std']
    
    X_features = X[feature_columns]
    X_scaled = scaler.transform(X_features)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return predictions, probabilities



st.markdown('<p class="main-header">🏃 Fitness Activity Classifier</p>', unsafe_allow_html=True)
st.markdown("### Classify physical activities using accelerometer data")
st.markdown("---")

if model is None or scaler is None:
    st.stop()


with st.sidebar:
    st.header("📊 Input Options")
    input_mode = st.radio(
        "Choose input method:",
        ["Manual Input", "Upload CSV", "Real-time Simulation"],
        help="Select how you want to provide accelerometer data"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app classifies physical activities:
    - 🚶 Walking
    - 🏃 Jogging  
    - 🪑 Sitting
    
    Using 3-axis accelerometer data (X, Y, Z).
    """)


if input_mode == "Manual Input":
    st.header(" Manual Input Mode")
    st.markdown("Enter accelerometer readings to classify the activity.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accel_x = st.number_input(
            "X-axis Acceleration (m/s²)",
            min_value=-20.0,
            max_value=20.0,
            value=0.5,
            step=0.1,
            help="Acceleration in X direction"
        )
    
    with col2:
        accel_y = st.number_input(
            "Y-axis Acceleration (m/s²)",
            min_value=-20.0,
            max_value=20.0,
            value=0.3,
            step=0.1,
            help="Acceleration in Y direction"
        )
    
    with col3:
        accel_z = st.number_input(
            "Z-axis Acceleration (m/s²)",
            min_value=-20.0,
            max_value=20.0,
            value=9.8,
            step=0.1,
            help="Acceleration in Z direction (gravity)"
        )
    
    if st.button(" Classify Activity", type="primary"):
        
        input_df = pd.DataFrame({
            'accel_x': [accel_x] * 10,  
            'accel_y': [accel_y] * 10,
            'accel_z': [accel_z] * 10
        })
        
        
        input_df = calculate_features(input_df)
        
        
        predictions, probabilities = predict_activity(input_df, model, scaler)
        predicted_activity = predictions[0]
        probs = probabilities[0]
        
        
        st.markdown("---")
        st.subheader(" Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            
            activity_emoji = {
                'walking': '🚶',
                'jogging': '🏃',
                'sitting': '🪑'
            }
            
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 1rem;'>
                <h1 style='font-size: 4rem; margin: 0;'>{activity_emoji.get(predicted_activity, '❓')}</h1>
                <h2 style='color: #1E88E5; margin: 0.5rem 0;'>{predicted_activity.upper()}</h2>
                <p style='font-size: 1.5rem; color: #666; margin: 0;'>{probs.max()*100:.1f}% confident</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            
            prob_df = pd.DataFrame({
                'Activity': ['Jogging', 'Sitting', 'Walking'],
                'Probability': probs * 100
            })
            
            fig = px.bar(
                prob_df,
                x='Probability',
                y='Activity',
                orientation='h',
                color='Probability',
                color_continuous_scale='Blues',
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                title="Probability Distribution",
                xaxis_title="Confidence (%)",
                yaxis_title="",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


elif input_mode == "Upload CSV":
    st.header("📁 Upload CSV File")
    st.markdown("Upload a CSV file with columns: `accel_x`, `accel_y`, `accel_z`")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV should contain accel_x, accel_y, accel_z columns"
    )
    
    if uploaded_file is not None:
        try:
            
            df = pd.read_csv(uploaded_file)
            
            st.success(f"  Loaded {len(df)} data points")
            
           
            with st.expander(" View Data Preview"):
                st.dataframe(df.head(20), use_container_width=True)
            
            if st.button(" Classify Activities", type="primary"):
                
                df_features = calculate_features(df.copy())
                
                if df_features is not None:
                    
                    predictions, probabilities = predict_activity(df_features, model, scaler)
                    
                    
                    df['predicted_activity'] = predictions
                    df['confidence'] = probabilities.max(axis=1) * 100
                    
                   
                    st.markdown("---")
                    st.subheader(" Classification Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Activity counts
                    activity_counts = df['predicted_activity'].value_counts()
                    
                    with col1:
                        st.metric("🚶 Walking", activity_counts.get('walking', 0))
                    with col2:
                        st.metric("🏃 Jogging", activity_counts.get('jogging', 0))
                    with col3:
                        st.metric("🪑 Sitting", activity_counts.get('sitting', 0))
                   
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        
                        fig = px.pie(
                            values=activity_counts.values,
                            names=activity_counts.index,
                            title="Activity Distribution",
                            color_discrete_sequence=px.colors.sequential.Blues_r
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                       
                        fig = go.Figure()
                        
                        for activity in df['predicted_activity'].unique():
                            activity_mask = df['predicted_activity'] == activity
                            fig.add_trace(go.Scatter(
                                x=df[activity_mask].index,
                                y=df[activity_mask]['accel_z'],
                                mode='markers',
                                name=activity,
                                marker=dict(size=5)
                            ))
                        
                        fig.update_layout(
                            title="Activity Timeline (Z-axis)",
                            xaxis_title="Sample",
                            yaxis_title="Z Acceleration (m/s²)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=" Download Results as CSV",
                        data=csv,
                        file_name=f"classified_activities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                   
                    with st.expander(" View Detailed Results"):
                        st.dataframe(df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


elif input_mode == "Real-time Simulation":
    st.header("  Real-time Activity Simulation")
    st.markdown("Simulate real-time accelerometer data for different activities.")
    
    activity_choice = st.selectbox(
        "Select Activity to Simulate",
        ["Walking", "Jogging", "Sitting"],
        help="Choose which activity to simulate"
    )
    
    num_samples = st.slider(
        "Number of Samples",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of data points to generate"
    )
    
    if st.button(" Start Simulation", type="primary"):
        # Generate simulated data
        activity_lower = activity_choice.lower()
        
        if activity_lower == 'walking':
            x = np.random.normal(0.5, 0.3, num_samples) + 0.3 * np.sin(np.linspace(0, 20*np.pi, num_samples))
            y = np.random.normal(0.3, 0.25, num_samples) + 0.2 * np.sin(np.linspace(0, 20*np.pi, num_samples))
            z = np.random.normal(9.8, 0.4, num_samples) + 0.5 * np.sin(np.linspace(0, 20*np.pi, num_samples))
        elif activity_lower == 'jogging':
            x = np.random.normal(1.0, 0.6, num_samples) + 0.8 * np.sin(np.linspace(0, 40*np.pi, num_samples))
            y = np.random.normal(0.5, 0.5, num_samples) + 0.6 * np.sin(np.linspace(0, 40*np.pi, num_samples))
            z = np.random.normal(9.8, 0.8, num_samples) + 1.2 * np.sin(np.linspace(0, 40*np.pi, num_samples))
        else:  # sitting
            x = np.random.normal(0.0, 0.1, num_samples)
            y = np.random.normal(0.0, 0.1, num_samples)
            z = np.random.normal(9.8, 0.15, num_samples)
        
        sim_df = pd.DataFrame({
            'accel_x': x,
            'accel_y': y,
            'accel_z': z
        })
        
      
        sim_df = calculate_features(sim_df)
        
       
        predictions, probabilities = predict_activity(sim_df, model, scaler)
        
       
        accuracy = (predictions == activity_lower).sum() / len(predictions) * 100
        
       
        st.markdown("---")
        st.subheader(" Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ground Truth", activity_choice)
        with col2:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col3:
            st.metric("Samples", num_samples)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=x, name='X-axis', mode='lines'))
        fig.add_trace(go.Scatter(y=y, name='Y-axis', mode='lines'))
        fig.add_trace(go.Scatter(y=z, name='Z-axis', mode='lines'))
        
        fig.update_layout(
            title=f"Simulated Accelerometer Data - {activity_choice}",
            xaxis_title="Sample",
            yaxis_title="Acceleration (m/s²)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        
        pred_counts = pd.Series(predictions).value_counts()
        fig = px.bar(
            x=pred_counts.index,
            y=pred_counts.values,
            labels={'x': 'Predicted Activity', 'y': 'Count'},
            title="Prediction Distribution",
            color=pred_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Streamlit | Fitness Activity Classifier</p>
</div>
""", unsafe_allow_html=True)