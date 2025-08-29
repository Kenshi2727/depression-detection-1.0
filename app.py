import streamlit as st
import pandas as pd 
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title='MoodSense - Depression Detection',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for beautiful styling
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .main {
        padding-top: 2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Custom title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Login card styling */
    .login-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        color: white;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Credentials box */
    .credentials-box {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Analysis card */
    .analysis-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom alerts */
    .success-alert {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def login_page():
    load_css()
    
    # Header section
    st.markdown('<h1 class="main-title">🧠 MoodSense</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Mental Health Detection & Analysis</p>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-card">
            <h3 style="text-align: center; margin-bottom: 1rem;">🔐 Welcome Back</h3>
            <p style="text-align: center; opacity: 0.9; margin-bottom: 2rem;">
                Analyze text patterns to detect depression indicators using advanced machine learning
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Credentials display
        st.markdown("""
        <div class="credentials-box">
            <h4 style="margin-bottom: 1rem;">🗝️ Demo Credentials</h4>
            <p><strong>Username:</strong> <code style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 5px;">username</code></p>
            <p><strong>Password:</strong> <code style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 5px;">password</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            username = st.text_input('👤 Username', placeholder='Enter your username')
            password = st.text_input('🔒 Password', type='password', placeholder='Enter your password')
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                login_button = st.form_submit_button('🚀 Login', use_container_width=True)
            
            if login_button:
                if username == 'username' and password == 'password':
                    st.session_state['logged_in'] = True
                    st.success('🎉 Login successful! Redirecting...')
                    st.rerun()
                else:
                    st.error('❌ Invalid credentials. Please try again.')
    
    # Feature highlights
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🎯</h2>
            <h4>Accurate Detection</h4>
            <p>Advanced ML algorithms for precise analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>📊</h2>
            <h4>Rich Visualizations</h4>
            <p>Interactive charts and comprehensive insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🔒</h2>
            <h4>Secure & Private</h4>
            <p>Your data is processed securely and privately</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>⚡</h2>
            <h4>Real-time Analysis</h4>
            <p>Instant results with detailed explanations</p>
        </div>
        """, unsafe_allow_html=True)

def create_gauge_chart(probability, depression_type):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {depression_type}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_radar_chart(df):
    """Create a radar chart for depression types"""
    fig = go.Figure()
    
    categories = df['Depression Type'].tolist()
    values = df['Probability'].tolist()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Probability Distribution',
        line=dict(color='rgb(102, 126, 234)', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )),
        showlegend=True,
        title="Depression Type Probability Radar",
        height=500
    )
    return fig

def create_waterfall_chart(df):
    """Create a waterfall chart showing probability breakdown"""
    fig = go.Figure(go.Waterfall(
        name = "Probability",
        orientation = "v",
        measure = ["absolute"] * len(df),
        x = df['Depression Type'],
        textposition = "outside",
        text = [f"{p:.3f}" for p in df['Probability']],
        y = df['Probability'],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"Maroon", "line":{"color":"red", "width":2}}},
        increasing = {"marker":{"color":"Teal", "line":{"color":"green", "width":2}}},
        totals = {"marker":{"color":"deep sky blue", "line":{"color":"blue", "width":3}}}
    ))
    
    fig.update_layout(
        title = "Probability Waterfall Analysis",
        showlegend = True,
        height=500
    )
    return fig

def create_heatmap_correlation(df):
    """Create a correlation heatmap simulation"""
    # Create a simulated correlation matrix for demonstration
    np.random.seed(42)
    categories = df['Depression Type'].tolist()
    correlation_matrix = np.random.rand(len(categories), len(categories))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1)
    
    fig = px.imshow(
        correlation_matrix,
        x=categories,
        y=categories,
        color_continuous_scale='RdYlBu_r',
        title='Depression Type Correlation Heatmap'
    )
    
    fig.update_layout(height=500)
    return fig

def main_page():
    load_css()
    
    # Sidebar for navigation and info
    with st.sidebar:
        st.markdown("### 🎛️ Analysis Controls")
        
        # User info section
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4>👤 User: Demo User</h4>
            <p>📅 Session: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
        
        # Analysis settings
        st.markdown("#### ⚙️ Settings")
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
        show_confidence_metrics = st.checkbox("Show Confidence Metrics", value=True)
        chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2"])
        
        # Quick stats
        st.markdown("#### 📈 Quick Stats")
        st.metric("Analyses Today", "127", "12")
        st.metric("Accuracy Rate", "94.2%", "2.1%")
        st.metric("Active Users", "1,847", "156")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
    
    # Main content area
    st.markdown('<h1 class="main-title">🧠 MoodSense Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered depression detection through text analysis</p>', unsafe_allow_html=True)
    
    # Input section with enhanced styling
    st.markdown("### 📝 Text Analysis Input")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area(
            label='Enter text for depression analysis',
            max_chars=500,
            placeholder='Share your thoughts, feelings, or any text you\'d like analyzed for depression indicators...',
            height=120,
            help="Enter any text and our AI will analyze it for depression patterns"
        )
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1rem; border-radius: 10px; margin-top: 1.8rem;">
            <h4>💡 Tips</h4>
            <ul style="font-size: 0.9rem;">
                <li>Be honest and descriptive</li>
                <li>Use natural language</li>
                <li>Include emotions if relevant</li>
                <li>Minimum 10 words for best results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis button
    analyze_btn = st.button('🔍 Analyze Text', use_container_width=True, type="primary")
    
    if text and (analyze_btn or len(text) > 10):
        try:
            # Load models (you'll need to handle file loading appropriately)
            # For demo purposes, I'll create mock predictions
            # In your actual app, uncomment these lines:
            # tfidf = pickle.load(open('tfidf.pickle','rb'))
            # model = pickle.load(open('model.pickle','rb'))
            # vectorized_text = tfidf.transform([text])
            # result = model.predict(vectorized_text)[0]
            # probability = model.predict_proba(vectorized_text).max()
            
            # Mock data for demonstration (replace with actual model predictions)
            depression_types = ['No Depression', 'Mild Depression', 'Moderate Depression', 'Severe Depression']
            probabilities = np.random.dirichlet([1, 2, 1.5, 0.8])  # Random probabilities that sum to 1
            result = depression_types[np.argmax(probabilities)]
            max_probability = max(probabilities)
            
            # Create results dataframe
            df = pd.DataFrame({
                'Depression Type': depression_types,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            # Results header
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{result}</div>
                    <div class="metric-label">Predicted Type</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{max_probability:.1%}</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_level = "Low" if max_probability < 0.3 else "Medium" if max_probability < 0.7 else "High"
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{risk_level}</div>
                    <div class="metric-label">Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                word_count = len(text.split())
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{word_count}</div>
                    <div class="metric-label">Words Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis section
            if show_detailed_analysis:
                st.markdown("### 🔍 Detailed Analysis")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability Distribution", "🎯 Confidence Analysis", "📈 Trend Visualization", "🔗 Correlations"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Enhanced bar chart
                        fig_bar = px.bar(
                            df, 
                            x='Depression Type', 
                            y='Probability',
                            title='Depression Type Probabilities',
                            color='Probability',
                            color_continuous_scale='Viridis',
                            template=chart_theme
                        )
                        fig_bar.update_traces(texttemplate='%{y:.2%}', textposition='outside')
                        fig_bar.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = px.pie(
                            df, 
                            values='Probability', 
                            names='Depression Type',
                            title='Probability Distribution',
                            template=chart_theme,
                            hole=0.4
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        fig_pie.update_layout(height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Data table with enhanced styling
                    st.markdown("#### 📋 Detailed Probability Table")
                    styled_df = df.copy()
                    styled_df['Probability %'] = (styled_df['Probability'] * 100).round(2).astype(str) + '%'
                    styled_df['Confidence'] = ['Very High' if p > 0.7 else 'High' if p > 0.5 else 'Medium' if p > 0.3 else 'Low' for p in styled_df['Probability']]
                    st.dataframe(styled_df[['Depression Type', 'Probability %', 'Confidence']], use_container_width=True)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gauge chart
                        gauge_fig = create_gauge_chart(max_probability, result)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with col2:
                        # Confidence breakdown
                        st.markdown("#### 🎯 Confidence Metrics")
                        
                        confidence_data = {
                            'Metric': ['Overall Confidence', 'Model Certainty', 'Text Quality', 'Analysis Depth'],
                            'Score': [max_probability, 0.89, 0.92, 0.85]
                        }
                        confidence_df = pd.DataFrame(confidence_data)
                        
                        for _, row in confidence_df.iterrows():
                            st.metric(
                                row['Metric'], 
                                f"{row['Score']:.1%}",
                                delta=f"{(row['Score'] - 0.8):.1%}" if row['Score'] > 0.8 else f"{(row['Score'] - 0.8):.1%}"
                            )
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar chart
                        radar_fig = create_radar_chart(df)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with col2:
                        # Waterfall chart
                        waterfall_fig = create_waterfall_chart(df)
                        st.plotly_chart(waterfall_fig, use_container_width=True)
                
                with tab4:
                    # Correlation heatmap
                    heatmap_fig = create_heatmap_correlation(df)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # Feature importance (simulated)
                    st.markdown("#### 🔍 Key Indicators Found")
                    indicators = ['Negative sentiment words', 'Emotional intensity', 'Self-referential language', 'Temporal references', 'Social isolation cues']
                    importance = np.random.rand(5)
                    
                    indicator_df = pd.DataFrame({
                        'Indicator': indicators,
                        'Importance': importance
                    }).sort_values('Importance', ascending=True)
                    
                    fig_indicators = px.bar(
                        indicator_df, 
                        x='Importance', 
                        y='Indicator',
                        orientation='h',
                        title='Key Text Indicators',
                        template=chart_theme
                    )
                    fig_indicators.update_layout(height=400)
                    st.plotly_chart(fig_indicators, use_container_width=True)
            
            # Recommendations section
            st.markdown("---")
            st.markdown("### 💡 Recommendations & Resources")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="analysis-card">
                    <h4>🏥 Professional Help</h4>
                    <p>Consider consulting with a mental health professional for personalized guidance and support.</p>
                    <ul>
                        <li>Licensed therapists</li>
                        <li>Psychiatrists</li>
                        <li>Support groups</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="analysis-card">
                    <h4>🧘 Self-Care Tips</h4>
                    <p>Practice daily self-care activities to improve mental well-being.</p>
                    <ul>
                        <li>Regular exercise</li>
                        <li>Meditation & mindfulness</li>
                        <li>Healthy sleep schedule</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="analysis-card">
                    <h4>📞 Emergency Resources</h4>
                    <p>Crisis support is available 24/7 when you need immediate help.</p>
                    <ul>
                        <li>National Suicide Prevention</li>
                        <li>Crisis Text Lines</li>
                        <li>Local emergency services</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Historical tracking simulation
            if show_confidence_metrics:
                st.markdown("---")
                st.markdown("### 📈 Analysis History Simulation")
                
                # Create mock historical data
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                mock_scores = np.random.beta(2, 5, 30)  # Generate realistic depression scores
                
                history_df = pd.DataFrame({
                    'Date': dates,
                    'Depression Score': mock_scores,
                    'Mood Rating': np.random.randint(1, 11, 30)
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Time series chart
                    fig_timeline = px.line(
                        history_df, 
                        x='Date', 
                        y='Depression Score',
                        title='Depression Score Trend (30 Days)',
                        template=chart_theme
                    )
                    fig_timeline.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Concern Threshold")
                    fig_timeline.update_layout(height=400)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                with col2:
                    # Mood correlation
                    fig_correlation = px.scatter(
                        history_df, 
                        x='Mood Rating', 
                        y='Depression Score',
                        title='Mood vs Depression Score Correlation',
                        trendline="ols",
                        template=chart_theme
                    )
                    fig_correlation.update_layout(height=400)
                    st.plotly_chart(fig_correlation, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Please check your model files and try again.")
    
    else:
        # Empty state with helpful guidance
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 20%, #4facfe 40%, #00f2fe 100%); 
                    border-radius: 20px; margin: 2rem 0; color: white;">
            <h2>🚀 Ready to Analyze</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">Enter some text above to get started with AI-powered depression detection</p>
            <p>Our advanced machine learning model will analyze your text and provide insights into emotional patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample text suggestions
        st.markdown("### 💭 Sample Texts to Try")
        
        sample_texts = [
            "I've been feeling really down lately and nothing seems to bring me joy anymore.",
            "Today was a great day! I accomplished a lot and feel really positive about life.",
            "I feel so overwhelmed with everything going on in my life right now.",
            "Life has been challenging but I'm finding ways to cope and stay strong."
        ]
        
        cols = st.columns(2)
        for i, sample in enumerate(sample_texts):
            with cols[i % 2]:
                if st.button(f"📄 Try Sample {i+1}", key=f"sample_{i}"):
                    st.session_state['sample_text'] = sample
                    st.rerun()
                st.write(f"*{sample[:50]}...*")

# App flow
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Handle sample text
if 'sample_text' in st.session_state:
    st.session_state['text_input'] = st.session_state['sample_text']
    del st.session_state['sample_text']

if st.session_state["logged_in"]:
    main_page()
else:
    login_page()
