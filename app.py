from flask import Flask,request,render_template
import pickle
import numpy as np # type: ignore
app=Flask(__name__)
loaded_model=pickle.load(open("penguin_species_dt.pkl","rb"))
island_encoder=pickle.load(open("island_encoder.pkl","rb"))
species_encoder=pickle.load(open("species_encoder.pkl","rb"))
gender_mapping={"male": 0, "female": 1}


@app.route("/")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get features from user input from the form
    bill_length_mm=float(request.form["bill_length_mm"])
    bill_depth_mm=float(request.form["bill_depth_mm"])
    flipper_length_mm=float(request.form["flipper_length_mm"])
    body_mass_g=float(request.form["body_mass_g"])
    year=int(request.form["year"])
    # Map the "sex" feature
    sex_input=request.form["sex"].lower()
    sex_encoded=gender_mapping.get(sex_input)
    # Encode "island" feature
    island_input=request.form["island"]
    island_encoded=island_encoder.transform([island_input])[0]
    features=np.array([[island_encoded,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex_encoded,year]])
    prediction=loaded_model.predict(features)

    species_mapping={0: "Adelie",1: "Chinstrap",2: "Gentoo"}
    predicted_species=species_mapping.get(prediction[0],"Unknown")
    return render_template("result.html",species=predicted_species)

if __name__=="__main__":
    app.run(debug=True)