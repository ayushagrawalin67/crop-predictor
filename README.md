# 🌱 Crop Prediction System

A Machine Learning based web application that predicts the most suitable crop to grow based on soil nutrients and environmental conditions.

---

## 🚀 Features

* 🔍 Predicts crop using ML model
* 🎨 Clean and modern UI (Flask + HTML/CSS)
* 🖼️ Dynamic crop images
* 📊 Crop information panel (soil, season, water)
* ⚡ Fast and user-friendly interface

---

## 🧠 Tech Stack

* Python 🐍
* Flask 🌐
* Scikit-learn 🤖
* HTML, CSS 🎨
* NumPy

---

## 📥 Input Parameters

The model takes the following inputs:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature (°C)
* Humidity (%)
* pH value
* Rainfall (mm)

---

## 📤 Output

* 🌾 Predicted Crop Name
* 🖼️ Crop Image (dynamic)
* 📊 Crop Details (optional)

---

## 🗂️ Project Structure

```
crop-predictor/
│
├── app.py
├── model.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── images/
```

---

## ⚙️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/crop-predictor.git
```

2. Navigate to project:

```
cd crop-predictor
```

3. Install dependencies:

```
pip install flask numpy scikit-learn
```

4. Run the app:

```
python app.py
```

5. Open in browser:

```
http://127.0.0.1:5000/
```

---

## 💡 Future Enhancements

* 🌦️ Weather API integration
* 📈 Crop yield prediction
* 💰 Profit estimation
* 📱 Mobile responsive improvements
* ☁️ Deployment on cloud (Render/Heroku)

---

## 👨‍💻 Author

Ayush Agrawal

---

## ⭐ If you like this project, give it a star!
