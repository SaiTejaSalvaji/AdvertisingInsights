
# 📊 Advertising Insights  
**Predict Sales Revenue from Advertising Investments**  

![GitHub repo size](https://img.shields.io/github/repo-size/SaiTejaSalvaji/AdvertisingInsights) ![GitHub last commit](https://img.shields.io/github/last-commit/SaiTejaSalvaji/AdvertisingInsights) ![GitHub stars](https://img.shields.io/github/stars/SaiTejaSalvaji/AdvertisingInsights?style=social)  

---

## 🚀 Overview  
**Advertising Insights** is a Streamlit-based web app that predicts sales revenue based on advertising investments in TV, radio, and newspapers. It provides insightful visualizations like scatter plots, pie charts, and heatmaps to help you understand the impact of different advertising channels on sales performance.

---

## 🎯 Features  
✅ Predict sales based on advertising investments (TV, Radio, Newspaper)  
✅ Interactive sidebar for input  
✅ Gauge chart for real-time prediction feedback  
✅ Dynamic bar and pie charts for investment breakdown  
✅ Correlation heatmap and pair plots for data analysis  
✅ Residual plots for model accuracy assessment  

---



## 🏗️ Tech Stack  
- **Python**  
- **Streamlit**  
- **Pandas**  
- **NumPy**  
- **Scikit-learn**  
- **Matplotlib**  
- **Plotly**  
- **Seaborn**  

---

## 📂 Project Structure  
```
├── advertising.csv         # Dataset used for training and prediction
├── company.jpg             # Logo for sidebar branding
├── project.joblib          # Trained Linear Regression model
├── app.py                  # Main Streamlit app file
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## 🏁 Installation  
### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/SaiTejaSalvaji/AdvertisingInsights.git
cd AdvertisingInsights
```

### 2️⃣ **Create a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate    # On MacOS/Linux
# OR
.\venv\Scripts\activate     # On Windows
```

### 3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage  
### ➡️ **Run the Streamlit App**  
```bash
streamlit run app.py
```

---

## 🗃️ Dataset  
The dataset (`advertising.csv`) contains:  
- **TV** – Advertising spend on TV (in ₹1000)  
- **Radio** – Advertising spend on Radio (in ₹1000)  
- **Newspaper** – Advertising spend on Newspaper (in ₹1000)  
- **Sales** – Actual sales generated (in ₹1000)  

---

## 🧠 Model  
- **Model Used:** Linear Regression (from Scikit-learn)  
- **Preprocessing:** Standard Scaler for data normalization  

---

## 📊 Visualizations  
✅ Gauge chart for predicted sales  
✅ Scatter plots for channel-wise spending vs. sales  
✅ Pie chart for percentage contribution  
✅ Heatmap for correlation between features  
✅ Residual plot for model accuracy  

---

## 💡 How It Works  
1. Load the advertising dataset  
2. Train a Linear Regression model  
3. Scale the data using `StandardScaler`  
4. Predict sales based on user inputs  
5. Display predictions and insights with dynamic charts  

---

## 🏆 Future Improvements  
- Add more complex models (e.g., Decision Trees, Random Forest)  
- Include more features like seasonality and market trends  
- Enhance UI with custom themes and dark mode  

---
## 👨‍💻 Author & Contact  
**Sai Teja Salvaji**  
- 🌐 [GitHub](https://github.com/SaiTejaSalvaji)  
- 📸 [Instagram](https://www.instagram.com/sai_teja26)  
- 💼 [LinkedIn](https://www.linkedin.com/in/sai-teja-rao-salvaji-9b9a15332)  

---

## ⭐ Support the Project  
If you find this project useful, consider supporting it in the following ways:  

✅ **Star the Repository** – If you found this helpful, give it a ⭐ on GitHub — it helps increase visibility.  
✅ **Fork and Contribute** – Found a bug or have an idea for improvement? Fork the repository, make changes, and submit a pull request.  
✅ **Report Issues** – If you encounter any issues, feel free to open an issue on GitHub — feedback is always welcome.  
✅ **Share the Project** – Help others discover this project by sharing it within your network.  
✅ **Follow for Updates** – Stay updated with new features and improvements by following me on GitHub.  

Your support and contributions are highly appreciated! 🙌  

---
