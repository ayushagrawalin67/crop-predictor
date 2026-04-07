import numpy as np
from flask import Flask, request, render_template
import os
import pickle

flask_app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates")
)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array([float_features])

    prediction = model.predict(features)
    raw_crop = str(prediction[0]).lower().strip()

    # search_term formatted for LoremFlickr (e.g. "muskmelon")
    search_term = raw_crop.replace(" ", "")

    # Reliable Image Service (LoremFlickr) - Pulls real photos
    image_url = f"https://loremflickr.com/600/400/{search_term}"


    # Database for all 22 crops
    crop_info = {
        "rice": {"season": "Kharif", "soil": "Clayey/Loamy", "water": "Very High"},
        "maize": {"season": "Kharif", "soil": "Alluvial/Loamy", "water": "Medium"},
        "chickpea": {"season": "Rabi", "soil": "Silt Loam", "water": "Low"},
        "kidneybeans": {"season": "Kharif/Rabi", "soil": "Deep Alluvial", "water": "Medium"},
        "pigeonpeas": {"season": "Kharif", "soil": "Black/Alluvial", "water": "Medium"},
        "mothbeans": {"season": "Kharif", "soil": "Light Sandy", "water": "Very Low"},
        "mungbean": {"season": "Summer/Kharif", "soil": "Loamy", "water": "Low"},
        "blackgram": {"season": "Kharif", "soil": "Loamy/Black", "water": "Low"},
        "lentil": {"season": "Rabi", "soil": "Alluvial", "water": "Low"},
        "pomegranate": {"season": "Annual", "soil": "Deep Sandy Loam", "water": "Medium"},
        "banana": {"season": "Annual", "soil": "Rich Alluvial", "water": "High"},
        "mango": {"season": "Summer", "soil": "Alluvial/Laterite", "water": "Medium"},
        "grapes": {"season": "Annual", "soil": "Well-drained Loamy", "water": "Medium"},
        "watermelon": {"season": "Summer", "soil": "Sandy/Sandy Loam", "water": "Medium"},
        "muskmelon": {"season": "Summer", "soil": "Sandy Loam", "water": "Medium"},
        "apple": {"season": "Annual/Spring", "soil": "Loamy/Silty", "water": "Medium"},
        "orange": {"season": "Annual", "soil": "Well-drained Alluvial", "water": "Medium"},
        "papaya": {"season": "Annual", "soil": "Sandy Loam", "water": "Medium"},
        "coconut": {"season": "Annual", "soil": "Coastal/Sandy", "water": "High"},
        "cotton": {"season": "Kharif", "soil": "Black Soil", "water": "Medium"},
        "jute": {"season": "Kharif", "soil": "Alluvial/Silt", "water": "Very High"},
        "coffee": {"season": "Annual", "soil": "Rich Loamy", "water": "High"}
    }

    info = crop_info.get(search_term, {"season": "Unknown", "soil": "Varied", "water": "Moderate"})

    return render_template(
        "result.html",
        crop=raw_crop.title(),
        image=image_url,
        info=info
    )


if __name__ == "__main__":
    flask_app.run(debug=True)