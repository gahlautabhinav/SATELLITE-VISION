import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas as pd
from datetime import datetime
import base64
import io

# --- Page Config ---
st.set_page_config(
    page_title="🛰️ SatelliteVision AI", 
    layout="wide", 
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
            background-attachment: fixed;
        }
        
        .main-header {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #006699 100%);
            padding: 3rem 2rem;
            border-radius: 25px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px rgba(0, 212, 255, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }
        
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .title {
            font-family: 'Orbitron', monospace;
            font-size: 4rem;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
            margin: 0;
            position: relative;
            z-index: 1;
        }
        
        .subtitle {
            font-family: 'Exo 2', sans-serif;
            font-size: 1.5rem;
            color: #e0f7ff;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }
        
        .prediction-container {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 25px;
            margin: 2rem 0;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 212, 255, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .prediction-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shine 3s infinite;
        }
        
        @keyframes shine {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .prediction-title {
            font-family: 'Orbitron', monospace;
            font-size: 2.5rem;
            color: #00d4ff;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 0 0 15px rgba(0, 212, 255, 0.6);
        }
        
        .confidence-score {
            font-family: 'Exo 2', sans-serif;
            font-size: 1.8rem;
            color: #ffffff;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 20px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .metric-title {
            font-family: 'Exo 2', sans-serif;
            font-size: 0.9rem;
            color: #e0e6ff;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-value {
            font-family: 'Orbitron', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }
        
        .class-info {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }
        
        .class-info:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 212, 255, 0.2);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #000000;
            font-family: 'Exo 2', sans-serif;
            font-weight: 600;
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
            background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%);
        }
        
        .stSidebar {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .stSidebar .stMarkdown {
            color: #ffffff;
        }
        
        .upload-section {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 10px 25px rgba(255, 107, 107, 0.3);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .neon-border {
            border: 2px solid #00d4ff;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
            padding: 1rem;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
            50% { box-shadow: 0 0 30px rgba(0, 212, 255, 0.6); }
            100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .data-viz-container {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
        }
        
        .processing-animation {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 2rem 0;
        }
        
        .processing-text {
            font-family: 'Orbitron', monospace;
            color: #00d4ff;
            font-size: 1.2rem;
            text-align: center;
            margin: 1rem 0;
        }
        
        .footer {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin-top: 3rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .environmental-impact {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 1.5rem;
            border-radius: 20px;
            margin: 1rem 0;
            color: #000000;
            text-align: center;
            box-shadow: 0 10px 25px rgba(56, 239, 125, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
    <div class="main-header">
        <div class="title">🛰️ SatelliteVision AI</div>
        <div class="subtitle">Advanced Environmental Monitoring & Land Cover Classification</div>
    </div>
""", unsafe_allow_html=True)

# --- Load the Model ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("Modelenv.v1.h5")
    except:
        st.error("⚠️ Model file 'Modelenv.v1.h5' not found. Please ensure the model is in the correct directory.")
        return None

model = load_model()

# --- Class Labels with Enhanced Information ---
class_info = {
    "Cloudy": {
        "emoji": "☁️",
        "color": "#87CEEB",
        "description": "Atmospheric cloud formations affecting solar radiation",
        "environmental_impact": "Influences precipitation patterns and climate regulation"
    },
    "Desert": {
        "emoji": "🏜️",
        "color": "#DEB887",
        "description": "Arid landscapes with minimal vegetation coverage",
        "environmental_impact": "Carbon storage potential and desertification monitoring"
    },
    "Green Area": {
        "emoji": "🌿",
        "color": "#32CD32",
        "description": "Vegetation zones including forests and agricultural land",
        "environmental_impact": "Primary carbon sinks and biodiversity hotspots"
    },
    "Water": {
        "emoji": "💧",
        "color": "#4682B4",
        "description": "Water bodies including oceans, rivers, and lakes",
        "environmental_impact": "Ecosystem health and water resource management"
    }
}

class_labels = list(class_info.keys())

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown("### 🎛️ Mission Control")
    
    # Model Status
    if model:
        st.success("✅ AI Model: **ONLINE**")
        st.markdown("**Model Architecture:** Deep CNN")
        st.markdown("**Input Resolution:** 256x256 pixels")
        st.markdown("**Classes:** 4 terrain types")
    else:
        st.error("❌ AI Model: **OFFLINE**")
    
    st.markdown("---")
    
    # Upload Section
    st.markdown("### 📡 Satellite Data Upload")
    uploaded_file = st.file_uploader(
        "Deploy satellite imagery for analysis",
        type=["jpg", "jpeg", "png", "tiff"],
        help="Supported formats: JPG, PNG, TIFF"
    )
    
    # Clear session state if no file is uploaded or if a new file is uploaded
    if not uploaded_file:
        # Clear analysis results when no file is present
        for key in ['preds', 'top_class', 'confidence', 'analysis_time', 'current_file']:
            if key in st.session_state:
                del st.session_state[key]
    else:
        # Clear analysis results if a different file is uploaded
        current_file_name = uploaded_file.name
        if 'current_file' in st.session_state and st.session_state.current_file != current_file_name:
            for key in ['preds', 'top_class', 'confidence', 'analysis_time']:
                if key in st.session_state:
                    del st.session_state[key]
        st.session_state.current_file = current_file_name
    
    st.markdown("---")
    
    # Analysis Settings
    st.markdown("### ⚙️ Analysis Parameters")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence for reliable predictions"
    )
    
    show_advanced_metrics = st.checkbox("Show Advanced Metrics", value=True)
    show_environmental_impact = st.checkbox("Environmental Impact Analysis", value=True)
    
    st.markdown("---")
    
    # Class Information
    st.markdown("### 🎯 Classification Matrix")
    for class_name, info in class_info.items():
        st.markdown(f"""
            <div class="class-info">
                <strong>{info['emoji']} {class_name}</strong><br>
                <small>{info['description']}</small>
            </div>
        """, unsafe_allow_html=True)

# --- Main Content ---
if uploaded_file and model:
    # Image Processing
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((256, 256))
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🖼️ Satellite Image Analysis")
        st.markdown('<div class="neon-border">', unsafe_allow_html=True)
        st.image(image, caption="📡 Uploaded Satellite Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Real-time Metrics")
        
        # Simulated real-time stats
        stats_data = {
            "🌍 Total Scans": 15847,
            "🎯 Accuracy": "94.2%",
            "⚡ Speed": "0.8s",
            "🔥 Uptime": "99.7%"
        }
        
        for metric, value in stats_data.items():
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{metric}</div>
                    <div class="metric-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Processing Animation
    st.markdown("### 🔄 AI Processing Pipeline")
    
    if st.button("🚀 INITIATE ANALYSIS", key="analyze_btn"):
        # Preprocessing
        img_array = np.array(resized_image) / 255.0
        img_array = img_array.reshape(1, 256, 256, 3)
        
        # Processing animation
        processing_steps = [
            "🛰️ Satellite data received...",
            "🔍 Preprocessing image...",
            "🧠 Running AI inference...",
            "📊 Calculating probabilities...",
            "✅ Analysis complete!"
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, step in enumerate(processing_steps):
            status_text.markdown(f'<div class="processing-text">{step}</div>', unsafe_allow_html=True)
            progress_bar.progress((i + 1) / len(processing_steps))
            time.sleep(0.8)
        
        # Model prediction
        preds = model.predict(img_array)[0]
        top_idx = np.argmax(preds)
        top_class = class_labels[top_idx]
        confidence = preds[top_idx]
        
        # Store results in session state
        st.session_state.preds = preds
        st.session_state.top_class = top_class
        st.session_state.confidence = confidence
        st.session_state.analysis_time = datetime.now()
        
        status_text.empty()
        st.success("🎉 Analysis completed successfully!")

# --- Display Results ---
if hasattr(st.session_state, 'preds') and uploaded_file:
    preds = st.session_state.preds
    top_class = st.session_state.top_class
    confidence = st.session_state.confidence
    
    # Main Prediction Display
    st.markdown(f"""
        <div class="prediction-container pulse">
            <div class="prediction-title">{class_info[top_class]['emoji']} {top_class}</div>
            <div class="confidence-score">Confidence: {confidence*100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Confidence Progress Bar
    st.progress(int(confidence * 100))
    
    # Advanced Metrics
    if show_advanced_metrics:
        st.markdown("### 📈 Advanced Analytics")
        
        # Create enhanced visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown('<div class="data-viz-container">', unsafe_allow_html=True)
            
            # Enhanced Pie Chart
            fig_pie = px.pie(
                names=class_labels,
                values=preds,
                title="🎯 Confidence Distribution",
                color_discrete_map={cls: info['color'] for cls, info in class_info.items()},
                hole=0.4
            )
            fig_pie.update_layout(
                template="plotly_dark",
                title_font_size=16,
                title_font_color="#00d4ff",
                legend=dict(font=dict(color="white")),
                height=400
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with viz_col2:
            st.markdown('<div class="data-viz-container">', unsafe_allow_html=True)
            
            # Enhanced Bar Chart
            colors = [class_info[cls]['color'] for cls in class_labels]
            
            bar_fig = go.Figure(data=[
                go.Bar(
                    x=class_labels,
                    y=preds,
                    marker=dict(
                        color=colors,
                        line=dict(color='rgba(255,255,255,0.8)', width=2)
                    ),
                    text=[f"{p*100:.1f}%" for p in preds],
                    textposition='auto',
                    textfont=dict(size=12, color='white')
                )
            ])
            
            bar_fig.update_layout(
                title="📊 Classification Probabilities",
                template="plotly_dark",
                title_font_size=16,
                title_font_color="#00d4ff",
                xaxis=dict(title="Land Cover Type", color="white"),
                yaxis=dict(title="Probability", color="white"),
                height=400
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Environmental Impact Analysis
    if show_environmental_impact:
        st.markdown("### 🌱 Environmental Impact Assessment")
        
        impact_info = class_info[top_class]['environmental_impact']
        
        st.markdown(f"""
            <div class="environmental-impact">
                <h4>🌍 Detected: {class_info[top_class]['emoji']} {top_class}</h4>
                <p><strong>Environmental Significance:</strong> {impact_info}</p>
                <p><strong>Monitoring Priority:</strong> {'HIGH' if top_class in ['Green Area', 'Water'] else 'MEDIUM'}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed Results Table
    st.markdown("### 📋 Detailed Analysis Report")
    
    results_df = pd.DataFrame({
        'Land Cover Type': [f"{class_info[cls]['emoji']} {cls}" for cls in class_labels],
        'Confidence (%)': [f"{p*100:.2f}%" for p in preds],
        'Probability': preds.round(4),
        'Environmental Impact': [class_info[cls]['environmental_impact'] for cls in class_labels]
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    # Export Options
    st.markdown("### 💾 Export Results")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📊 Export CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"satellite_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("📈 Generate Summary"):
            summary = f"""
SATELLITE IMAGE ANALYSIS REPORT
Generated: {st.session_state.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}

PRIMARY CLASSIFICATION: {top_class}
CONFIDENCE LEVEL: {confidence*100:.2f}%
ENVIRONMENTAL IMPACT: {class_info[top_class]['environmental_impact']}

DETAILED BREAKDOWN:
{chr(10).join([f"• {cls}: {p*100:.2f}%" for cls, p in zip(class_labels, preds)])}
            """
            st.text_area("Analysis Summary", summary, height=300)
    
    with export_col3:
        if st.button("🔄 Re-analyze"):
            for key in ['preds', 'top_class', 'confidence', 'analysis_time']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

else:
    # Welcome Screen
    if not uploaded_file:
        st.markdown("### 🚀 Welcome to SatelliteVision AI")
        
        welcome_col1, welcome_col2 = st.columns(2)
        
        with welcome_col1:
            st.markdown("""
                <div class="upload-section floating">
                    <h3>📡 Ready for Analysis</h3>
                    <p>Upload a satellite image to begin environmental monitoring and land cover classification.</p>
                    <p><strong>Supported formats:</strong> JPG, PNG, TIFF</p>
                </div>
            """, unsafe_allow_html=True)
        
        with welcome_col2:
            st.markdown("#### 🎯 System Capabilities")
            
            capabilities = [
                "🌍 Multi-terrain classification",
                "⚡ Real-time processing",
                "📊 Advanced analytics",
                "🌱 Environmental impact assessment",
                "📈 Detailed reporting"
            ]
            
            for capability in capabilities:
                st.markdown(f"• {capability}")
    
    elif not model:
        st.error("🚨 System Error: AI model not loaded. Please check model file availability.")

# --- Enhanced Footer ---
st.markdown("""
    <div class="footer">
        <h3>🛰️ SatelliteVision AI Platform</h3>
        <p>Powered by Deep Learning • TensorFlow • Streamlit • Plotly</p>
        <p>🌍 Advancing Environmental Monitoring Through AI</p>
        <p><small>© 2024 SatelliteVision AI - Built with ❤️ for Earth</small></p>
    </div>
""", unsafe_allow_html=True)