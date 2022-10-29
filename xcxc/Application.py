from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('mushrooms_rf.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        bruises = int(request.form['bruises'])
        odor = int(request.form['odor'])
        gill_spacing = int(request.form['gill_spacing'])
        gill_size = int(request.form['gill_size'])
        gill_color = int(request.form['gill_color'])
        stalk_shape = int(request.form['stalk_shape'])
        stalk_root = int(request.form['stalk_root'])
        gill_spacing = int(request.form['gill_spacing'])
        stalk_surface_above_ring = int(request.form['stalk_surface_above_ring'])
        stalk_surface_below_ring = int(request.form['stalk_surface_below_ring'])
        stalk_color_above_ring = int(request.form['stalk_color_above_ring'])
        ring_type = int(request.form['ring_type'])
        spore_print_color = int(request.form['spore_print_color'])
        population = int(request.form['population'])
        habitat = int(request.form['habitat'])
        
    values = np.array([[bruises,odor,
                        gill_spacing,gill_size,gill_color,stalk_shape,stalk_root,
                        stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,ring_type,
                        spore_print_color,population,habitat]])

    prediction = model.predict(values)
    if prediction == 0:
        label = 'Edible'
    else:
        label = 'Poisonous'

    return render_template('result.html', prediction_text='Mushroom is {}'.format(label))





if __name__ == "__main__":
    app.run(debug=True)

