# 🏠 House Price Estimator - How to Run

## Prerequisites

First, install the required dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

## Option 1: 🖥️ Terminal/Command Line (Original)

Run the original `deploy.py` script:

```bash
python3 deploy.py
```

**Output:** Displays a single prediction in the terminal with the sample data.

**Pros:** Simple, no extra dependencies  
**Cons:** Not interactive, only shows one example

---

## Option 2: 🌐 Streamlit Web App (RECOMMENDED - Most Beautiful!)

### Install Streamlit:
```bash
pip install streamlit
```

### Run the app:
```bash
streamlit run app_streamlit.py
```

**What happens:**
- Opens a beautiful web interface in your browser automatically
- Interactive sliders, dropdowns, and input fields
- Real-time predictions with nice formatting
- Professional-looking UI

**Pros:** 
- Most visually appealing
- Easy to use with nice UI components
- Great for demos and presentations

**Cons:** 
- Requires Streamlit installation

---

## Option 3: 🎯 Gradio Interface (Also Great!)

### Install Gradio:
```bash
pip install gradio
```

### Run the app:
```bash
python3 app_gradio.py
```

**What happens:**
- Opens a web interface with a clean design
- Includes example inputs you can try
- Generates a shareable public link (optional)

**Pros:**
- Very quick to set up
- Can create shareable links
- Simple and clean interface

**Cons:**
- Requires Gradio installation

---

## Comparison Table

| Method | Visual | Interactive | Easy Setup | Best For |
|--------|--------|-------------|------------|----------|
| Terminal (`deploy.py`) | ❌ | ❌ | ✅ | Quick testing |
| Streamlit (`app_streamlit.py`) | ✅✅✅ | ✅ | ✅ | Presentations, demos |
| Gradio (`app_gradio.py`) | ✅✅ | ✅ | ✅ | Quick prototypes, sharing |

---

## Troubleshooting

### "Module not found" error?
Install missing packages:
```bash
pip install pandas numpy scikit-learn pickle5
```

### "model.pkl not found" error?
Make sure `model.pkl` and `scaler.pkl` are in the same directory as the script.

### Port already in use?
For Streamlit:
```bash
streamlit run app_streamlit.py --server.port 8502
```

For Gradio:
```bash
# Edit the last line in app_gradio.py to:
demo.launch(server_port=7861)
```

---

## 🎉 Quick Start Recommendation

If you want the **most visual experience**, use Streamlit:

```bash
pip install streamlit pandas numpy scikit-learn
streamlit run app_streamlit.py
```

Your browser will open automatically with a beautiful interface! 🚀


