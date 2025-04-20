# workflow-ai

# Create environment (optional)
conda create -n pr-insights python=3.10 -y
conda activate pr-insights

# Install dependencies
pip install -r requirements.txt

streamlit run app.py


ai-productivity-dashboard/
├── app.py
├── chat_session.log
├── insights_summary.txt
├── Pull_Request_Data.csv
├── requirements.txt
└── README.md
