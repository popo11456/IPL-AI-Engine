import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Set up the App Page
st.set_page_config(page_title="IPL AI Engine", layout="wide")
st.title("🏏 IPL Win Probability & Team Analytics Engine")

# Load your 103MB Data
@st.cache_data
def load_data():
    # Use your specific path here
  df = pd.read_csv('IPL_small.csv', low_memory=False)
    return df

try:
    df = load_data()
    all_teams = sorted(df['batting_team'].unique().astype(str))

    # --- SIDEBAR FOR INPUTS ---
    st.sidebar.header("Match Selection")
    team1 = st.sidebar.selectbox("Select Batting Team", all_teams)
    team2 = st.sidebar.selectbox("Select Bowling Team", all_teams)
    venue = st.sidebar.selectbox("Select Venue", sorted(df['venue'].unique().astype(str)))

    # --- AI MODEL LOGIC ---
    # Simple training on the fly for the project demo
    le = LabelEncoder()
    le.fit(all_teams)
    
    # Filter for valid match results
    train_df = df[df['match_won_by'].isin(all_teams)].copy()
    X = train_df[['batting_team', 'bowling_team']].apply(le.transform)
    y = le.transform(train_df['match_won_by'])

    model = LogisticRegression()
    model.fit(X, y)

    # --- PREDICTION BUTTON ---
    if st.sidebar.button("Calculate Win Probability"):
        t1_idx = le.transform([team1])[0]
        t2_idx = le.transform([team2])[0]
        
        # Get probability
        probs = model.predict_proba([[t1_idx, t2_idx]])[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=team1, value=f"{probs[0]*100:.1f}%")
        with col2:
            st.metric(label=team2, value=f"{probs[1]*100:.1f}%")
        
        st.progress(probs[0]) # Visual bar showing the lead

    # --- ANALYTICS SECTION ---
    st.header("📊 Historical Team Analytics")
    # Show a chart of total wins for the selected teams
    win_counts = df[df['match_won_by'].isin([team1, team2])]['match_won_by'].value_counts()
    
    fig, ax = plt.subplots()
    win_counts.plot(kind='bar', ax=ax, color=['#3498db', '#f1c40f'])
    plt.title("Head-to-Head Total Wins")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading data: {e}")

    st.info("Make sure your IPL.csv is on your Desktop!")
