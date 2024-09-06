from flask import Flask, render_template, request, jsonify
from MultilingualTextToImageGenerator import generate_image  # Assuming this is your module for image generation

app = Flask(__name__)

# Route for serving the generate page
@app.route('/')
def home():
    return render_template('generate.html')

# Route to handle image generation requests
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')  # Get the prompt from the form
    try:
        # Assuming generate_image is the function that generates the image based on the input prompt
        image_url = generate_image(prompt)  # This should return the URL of the generated image
        return jsonify({'status': 'success', 'image_url': image_url})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
