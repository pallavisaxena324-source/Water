from flask import Flask, render_template, request
from main import predict_water_quality

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        input_data = {
            'Temp': float(request.form['temp']),
            'D.O. (mg/l)': float(request.form['do']),
            'PH': float(request.form['ph']),
            'CONDUCTIVITY (µmhos/cm)': float(request.form['conductivity']),
            'B.O.D. (mg/l)': float(request.form['bod'])
        }
        
        # Get prediction
        prediction = predict_water_quality(input_data)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)