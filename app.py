from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('model/color_match_model.pkl')
label_encoder_X = joblib.load('model/label_encoder_X.pkl')
label_encoder_y = joblib.load('model/label_encoder_y.pkl')

# Load the dataset for reference
df = pd.read_csv('Book1.csv')

@app.route('/', methods=['GET', 'POST'])
def color_match():
    if request.method == 'POST':
        color = request.form.get('color').capitalize().strip()  # Capitalize and strip spaces
        
        # Check if color is in the dataset
        if color not in df['Color'].values:
            return jsonify({'best_match': 'Color not found in dataset'})

        # Encode the input color
        color_encoded = label_encoder_X.transform([color])

        try:
            # Predict the best match using the model
            best_match_encoded = model.predict([[color_encoded[0]]])
            best_match = label_encoder_y.inverse_transform(best_match_encoded)
            return jsonify({'best_match': best_match[0]})
        except Exception as e:
            return jsonify({'best_match': f'Error: {str(e)}'})
    
    return render_template('color_match.html')

if __name__ == '__main__':
    app.run(debug=True)
