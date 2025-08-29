# (Full file — same improved UI code as before, but with rerun calls removed and replaced by safe returns.)
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

# Custom CSS for beautiful styling & better alignment
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Page header */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0.25rem 0 0.25rem 0;
    }

    .subtitle {
        font-size: 1rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }

    /* Card wrappers */
    .card {
        background: white;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 8px 20px rgba(16,24,40,0.06);
        border: 1px solid rgba(15,23,42,0.04);
    }

    .login-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        margin: 0.75rem 0;
    }

    .credentials-box {
        background: rgba(255,255,255,0.06);
        padding: 0.9rem;
        border-radius: 10px;
        color: white;
        margin-top: 0.6rem;
    }

    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
        min-height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .metric-container:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 40px rgba(14,30,60,0.12);
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
        font-weight: 500;
    }

    /* Chart card */
    .chart-card {
        background: white;
        padding: 0.65rem;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(12,34,63,0.06);
        border: 1px solid rgba(15,23,42,0.04);
        margin-bottom: 0.9rem;
    }

    /* Tips box */
    .tips-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.9rem;
        border-radius: 10px;
        margin-top: 0.4rem;
    }

    /* Sidebar custom wrapper (use inside st.sidebar with unsafe HTML) */
    .sidebar-card {
        background: rgba(255,255,255,0.06);
        padding: 0.75rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
    }

    /* Small responsive tweaks */
    @media (max-width: 800px) {
        .main-title { font-size: 1.8rem; }
        .metric-value { font-size: 1.05rem; }
    }

    /* Hide default Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# -------- Plotly helper functions (accept chart_theme) --------
def create_gauge_chart(probability, depression_type, chart_theme=None):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {depression_type}"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(height=360)
    if chart_theme:
        fig.update_layout(template=chart_theme)
    return fig

def create_radar_chart(df, chart_theme=None):
    """Create a radar chart for depression types"""
    fig = go.Figure()
    categories = df['Depression Type'].tolist()
    values = df['Probability'].tolist()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Probability Distribution',
        line=dict(color='rgb(102, 126, 234)', width=2.5),
        fillcolor='rgba(102, 126, 234, 0.22)'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.1])),
        showlegend=False,
        title="Depression Type Probability Radar",
        height=420
    )
    if chart_theme:
        fig.update_layout(template=chart_theme)
    return fig

def create_waterfall_chart(df, chart_theme=None):
    """Create a waterfall chart showing probability breakdown"""
    fig = go.Figure(go.Waterfall(
        name="Probability",
        orientation="v",
        measure=["absolute"] * len(df),
        x=df['Depression Type'],
        textposition="outside",
        text=[f"{p:.3f}" for p in df['Probability']],
        y=df['Probability'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "Maroon", "line": {"color": "red", "width": 2}}},
        increasing={"marker": {"color": "Teal", "line": {"color": "green", "width": 2}}},
        totals={"marker": {"color": "deepskyblue", "line": {"color": "blue", "width": 3}}}
    ))
    fig.update_layout(title="Probability Waterfall Analysis", showlegend=False, height=420)
    if chart_theme:
        fig.update_layout(template=chart_theme)
    return fig

def create_heatmap_correlation(df, chart_theme=None):
    """Create a correlation heatmap simulation"""
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
    fig.update_layout(height=420)
    if chart_theme:
        fig.update_layout(template=chart_theme)
    return fig

# ---------------- UI pages ----------------
def login_page():
    load_css()
    st.markdown('<h1 class="main-title">🧠 MoodSense</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Mental Health Detection & Analysis</p>', unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align:center;margin:0.15rem 0;">🔐 Welcome Back</h3>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;margin:0.25rem 0 0.6rem 0;opacity:0.95;">Analyze text patterns to detect depression indicators using advanced ML</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Credentials & form
        st.markdown('<div class="credentials-box">', unsafe_allow_html=True)
        st.markdown('<strong>🗝️ Demo Credentials</strong>', unsafe_allow_html=True)
        st.markdown('<p style="margin:6px 0;"><strong>Username:</strong> <code style="background: rgba(255,255,255,0.12); padding: 0.2rem 0.5rem; border-radius:4px;">username</code></p>', unsafe_allow_html=True)
        st.markdown('<p style="margin:6px 0;"><strong>Password:</strong> <code style="background: rgba(255,255,255,0.12); padding: 0.2rem 0.5rem; border-radius:4px;">password</code></p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")  # spacing
        with st.form("login_form"):
            u_col, p_col = st.columns([1, 1])
            with u_col:
                username = st.text_input('👤 Username', placeholder='Enter username')
            with p_col:
                password = st.text_input('🔒 Password', type='password', placeholder='Enter password')

            st.write("")  # spacing
            btn_col_left, btn_col_center, btn_col_right = st.columns([1, 1, 1])
            with btn_col_center:
                login_button = st.form_submit_button('🚀 Login', use_container_width=True)

            if login_button:
                if username == 'username' and password == 'password':
                    st.session_state['logged_in'] = True
                    st.success('🎉 Login successful!')
                    # NO rerun call — return and let Streamlit re-run naturally
                    return
                else:
                    st.error('❌ Invalid credentials. Please try again.')

    # Feature highlights (balanced)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.markdown('<div class="card" style="text-align:center;"><h3>🎯</h3><h4 style="margin:6px 0;">Accurate Detection</h4><p style="margin:0;font-size:0.9rem;color:#515151">Advanced ML algorithms for precise analysis</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card" style="text-align:center;"><h3>📊</h3><h4 style="margin:6px 0;">Rich Visualizations</h4><p style="margin:0;font-size:0.9rem;color:#515151">Interactive charts & insights</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card" style="text-align:center;"><h3>🔒</h3><h4 style="margin:6px 0;">Secure & Private</h4><p style="margin:0;font-size:0.9rem;color:#515151">Your data is processed locally</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card" style="text-align:center;"><h3>⚡</h3><h4 style="margin:6px 0;">Real-time Analysis</h4><p style="margin:0;font-size:0.9rem;color:#515151">Fast results with explanations</p></div>', unsafe_allow_html=True)

def main_page():
    load_css()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Analysis Controls")
        st.markdown('</div>', unsafe_allow_html=True)

        # user info card
        st.markdown(f'<div class="sidebar-card"><strong>👤 User:</strong> Demo User<br><small>Session: {datetime.now().strftime("%Y-%m-%d %H:%M")}</small></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card"><strong>⚙️ Settings</strong></div>', unsafe_allow_html=True)
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
        show_confidence_metrics = st.checkbox("Show Confidence Metrics", value=True)
        chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2"])

        st.markdown('<div class="sidebar-card"><strong>📈 Quick Stats</strong></div>', unsafe_allow_html=True)
        st.metric("Analyses Today", "127", "12")
        st.metric("Accuracy Rate", "94.2%", "2.1%")
        st.metric("Active Users", "1,847", "156")

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            # NO rerun call; return to stop the current render (Streamlit will re-run)
            return

    # Header
    st.markdown('<h1 class="main-title">🧠 MoodSense Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered depression detection through text analysis</p>', unsafe_allow_html=True)

    # Input area
    st.markdown("### 📝 Text Analysis Input")
    left_col, right_col = st.columns([3, 1], gap="large")
    with left_col:
        if 'text_input' not in st.session_state:
            st.session_state['text_input'] = ""
        text = st.text_area(
            label='',
            value=st.session_state.get('text_input', ''),
            max_chars=500,
            placeholder='Share your thoughts... (min ~10 words recommended)',
            height=140,
            help="Enter any text and our AI will analyze it for depression patterns"
        )
        st.session_state['text_input'] = text

    with right_col:
        st.markdown('<div class="tips-box"><h4 style="margin:0 0 6px 0;">💡 Tips</h4><ul style="margin:0 0 0 1rem;padding-left:0.4rem;"><li>Be honest and descriptive</li><li>Use natural language</li><li>Include emotions if relevant</li><li>~10+ words for best results</li></ul></div>', unsafe_allow_html=True)

    analyze_btn = st.button('🔍 Analyze Text', use_container_width=True)

    # If text present and button pressed or long text: run analysis
    if text and (analyze_btn or len(text.split()) > 10):
        try:
            # ----- Replace these mocks with your actual model loading -----
            # tfidf = pickle.load(open('tfidf.pickle','rb'))
            # model = pickle.load(open('model.pickle','rb'))
            # vectorized_text = tfidf.transform([text])
            # result = model.predict(vectorized_text)[0]
            # max_probability = model.predict_proba(vectorized_text).max()

            depression_types = ['No Depression', 'Mild Depression', 'Moderate Depression', 'Severe Depression']
            probabilities = np.random.dirichlet([1, 2, 1.5, 0.8])
            result = depression_types[np.argmax(probabilities)]
            max_probability = max(probabilities)

            df = pd.DataFrame({'Depression Type': depression_types, 'Probability': probabilities}).sort_values('Probability', ascending=False)

            # Results header
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # Metrics (aligned & consistent)
            m1, m2, m3, m4 = st.columns(4, gap="large")
            with m1:
                st.markdown(f'<div class="metric-container"><div class="metric-value">{result}</div><div class="metric-label">Predicted Type</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-container"><div class="metric-value">{max_probability:.1%}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
            with m3:
                risk_level = "Low" if max_probability < 0.3 else "Medium" if max_probability < 0.7 else "High"
                st.markdown(f'<div class="metric-container"><div class="metric-value">{risk_level}</div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)
            with m4:
                word_count = len(text.split())
                st.markdown(f'<div class="metric-container"><div class="metric-value">{word_count}</div><div class="metric-label">Words Analyzed</div></div>', unsafe_allow_html=True)

            # Detailed analysis as tabs
            if show_detailed_analysis:
                st.markdown("### 🔍 Detailed Analysis")
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability Distribution", "🎯 Confidence Analysis", "📈 Trend Visualization", "🔗 Correlations"])

                # Tab 1 - distribution + table
                with tab1:
                    left, right = st.columns([2, 1], gap="large")
                    with left:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
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
                        fig_bar.update_layout(height=380)
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with right:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                        fig_pie = px.pie(df, values='Probability', names='Depression Type', title='Probability Distribution', template=chart_theme, hole=0.38)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        fig_pie.update_layout(height=380)
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("#### 📋 Detailed Probability Table")
                    styled_df = df.copy()
                    styled_df['Probability %'] = (styled_df['Probability'] * 100).round(2).astype(str) + '%'
                    styled_df['Confidence'] = ['Very High' if p > 0.7 else 'High' if p > 0.5 else 'Medium' if p > 0.3 else 'Low' for p in styled_df['Probability']]
                    st.dataframe(styled_df[['Depression Type', 'Probability %', 'Confidence']], use_container_width=True)

                # Tab 2 - gauge + confidence metrics
                with tab2:
                    c1, c2 = st.columns([1, 1], gap="large")
                    with c1:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                        gauge_fig = create_gauge_chart(max_probability, result, chart_theme=chart_theme)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown("#### 🎯 Confidence Metrics")
                        confidence_data = {
                            'Metric': ['Overall Confidence', 'Model Certainty', 'Text Quality', 'Analysis Depth'],
                            'Score': [max_probability, 0.89, 0.92, 0.85]
                        }
                        confidence_df = pd.DataFrame(confidence_data)
                        for _, row in confidence_df.iterrows():
                            st.metric(row['Metric'], f"{row['Score']:.1%}", delta=f"{(row['Score'] - 0.8):.1%}")

                # Tab 3 - radar + waterfall
                with tab3:
                    c1, c2 = st.columns(2, gap="large")
                    with c1:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                        radar_fig = create_radar_chart(df, chart_theme=chart_theme)
                        st.plotly_chart(radar_fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                        waterfall_fig = create_waterfall_chart(df, chart_theme=chart_theme)
                        st.plotly_chart(waterfall_fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                # Tab 4 - correlations + indicators
                with tab4:
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    heatmap_fig = create_heatmap_correlation(df, chart_theme=chart_theme)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("#### 🔍 Key Indicators Found")
                    indicators = ['Negative sentiment words', 'Emotional intensity', 'Self-referential language', 'Temporal references', 'Social isolation cues']
                    importance = np.random.rand(5)
                    indicator_df = pd.DataFrame({'Indicator': indicators, 'Importance': importance}).sort_values('Importance', ascending=True)
                    fig_indicators = px.bar(indicator_df, x='Importance', y='Indicator', orientation='h', title='Key Text Indicators', template=chart_theme)
                    fig_indicators.update_layout(height=360)
                    st.plotly_chart(fig_indicators, use_container_width=True)

            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations & Resources")
            r1, r2, r3 = st.columns(3, gap="large")
            with r1:
                st.markdown('<div class="card"><h4 style="margin:0 0 6px 0;">🏥 Professional Help</h4><p style="margin:0;font-size:0.95rem">Consider consulting a mental health professional for personalized guidance.</p><ul style="margin:6px 0 0 1rem;"><li>Licensed therapists</li><li>Psychiatrists</li><li>Support groups</li></ul></div>', unsafe_allow_html=True)
            with r2:
                st.markdown('<div class="card"><h4 style="margin:0 0 6px 0;">🧘 Self-Care Tips</h4><p style="margin:0;font-size:0.95rem">Practice daily self-care to improve mental well-being.</p><ul style="margin:6px 0 0 1rem;"><li>Exercise</li><li>Meditation</li><li>Healthy sleep</li></ul></div>', unsafe_allow_html=True)
            with r3:
                st.markdown('<div class="card"><h4 style="margin:0 0 6px 0;">📞 Emergency Resources</h4><p style="margin:0;font-size:0.95rem">Crisis support is available 24/7.</p><ul style="margin:6px 0 0 1rem;"><li>National Suicide Prevention</li><li>Crisis Text Lines</li><li>Local emergency services</li></ul></div>', unsafe_allow_html=True)

            # Historical tracking (optional)
            if show_confidence_metrics:
                st.markdown("---")
                st.markdown("### 📈 Analysis History Simulation")
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                mock_scores = np.random.beta(2, 5, 30)
                history_df = pd.DataFrame({'Date': dates, 'Depression Score': mock_scores, 'Mood Rating': np.random.randint(1, 11, 30)})
                t1, t2 = st.columns(2, gap="large")
                with t1:
                    fig_timeline = px.line(history_df, x='Date', y='Depression Score', title='Depression Score Trend (30 Days)', template=chart_theme)
                    fig_timeline.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Concern Threshold")
                    fig_timeline.update_layout(height=380)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                with t2:
                    # NOTE: px.scatter trendline uses statsmodels (ensure statsmodels is installed if you want trendline="ols")
                    fig_correlation = px.scatter(history_df, x='Mood Rating', y='Depression Score', title='Mood vs Depression Score Correlation', trendline="ols", template=chart_theme)
                    fig_correlation.update_layout(height=380)
                    st.plotly_chart(fig_correlation, use_container_width=True)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Please check your model files and try again.")

    else:
        # Empty state / hero card
        st.markdown("""
        <div style="text-align:center;padding:1.2rem;border-radius:12px;margin:0.8rem 0;background:linear-gradient(135deg,#f093fb 0%,#f5576c 20%,#4facfe 60%);color:white">
            <h3 style="margin:0;">🚀 Ready to Analyze</h3>
            <p style="margin:6px 0 0 0;">Enter some text above to get started with AI-powered depression detection.</p>
        </div>
        """, unsafe_allow_html=True)

        # Sample texts
        st.markdown("### 💭 Sample Texts to Try")
        sample_texts = [
            "I've been feeling really down lately and nothing seems to bring me joy anymore.",
            "Today was a great day! I accomplished a lot and feel really positive about life.",
            "I feel so overwhelmed with everything going on in my life right now.",
            "Life has been challenging but I'm finding ways to cope and stay strong."
        ]
        c1, c2 = st.columns(2, gap="large")
        cols = [c1, c2]
        for i, sample in enumerate(sample_texts):
            with cols[i % 2]:
                if st.button(f"📄 Try Sample {i+1}", key=f"sample_{i}"):
                    st.session_state['text_input'] = sample
                    # NO rerun call — return to stop current rendering; Streamlit will re-run automatically
                    return
                st.write(f"*{sample[:70]}...*")

# ---------------- App flow ----------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Keep sample_text if set earlier
if 'sample_text' in st.session_state and 'text_input' not in st.session_state:
    st.session_state['text_input'] = st.session_state['sample_text']
    del st.session_state['sample_text']

if st.session_state["logged_in"]:
    main_page()
else:
    login_page()
