# Smart SMS Insights Dashboard

## How to Run the Code

1. Open your project folder (where this script is saved).

2. Ensure the following model files and directories exist:
   - `models/fine_tuned_embedding/`
   - `models/classifier_model.pt`
   - `models/sms_summary_model.pkl`

3. Install the required dependencies:
   ```bash
   pip install streamlit torch sentence-transformers pandas scikit-learn joblib
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. When the app starts, it will automatically open in your default web browser.  
   If you have deployed the app online (for example, on Streamlit Cloud), simply open your deployed link to access the dashboard.

---

## How to Use

1. Select a theme (Light or Dark) from the sidebar.  
2. Choose an input mode:
   - **Upload CSV File:** Upload a CSV containing a column named "Message" or similar.  
   - **Write / Paste Messages:** Enter one message per line in the format `Sender: Message`.  
3. Click **Analyze Messages** to process and summarize your data.  
4. The dashboard will display:
   - A category distribution chart.  
   - Summaries for each sender or message cluster.  
   - A detailed message table.  
5. Use the **Download Categorized Messages** button to export the analyzed data as a CSV file.
