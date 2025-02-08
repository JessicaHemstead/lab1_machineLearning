from flask import Flask, request, render_template_string, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

app = Flask(__name__)

def predictor_model(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    output = model.config.id2label[predicted_class_id]
    return output

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>String Predictor App</title>
</head>
<body>
    <h2>String value classifier</h2>
    <h3>Enter a String</h3>
    <form method="post">
        <input type="text" name="user_input" required>
        <button type="submit">Submit</button>
    </form>
    {% if result %}
        <h3>Input String: {{ user_string }}</h3>
        <h3>Predicted Output: {{ result }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_string = None
    if request.method == 'POST':
        user_string = request.form.get('user_input', '')
        result = predictor_model(user_string)
    return render_template_string(HTML_TEMPLATE, result=result, user_string=user_string)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('user_input', '')
    result = predictor_model(user_input)
    return jsonify({"input": user_input, "prediction": result})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')


