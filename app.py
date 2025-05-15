from flask import Flask,render_template, request, jsonify
import ollama
import cv2
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    image = request.files.get('frame')
    if not image:
        return jsonify({'error': 'No frame provided'}), 400

    # Save the uploaded frame to a temporary file
    temp_image_path = os.path.join('uploads', image.filename)
    image.save(temp_image_path)

    # Use Ollama to analyze the image
    response = ollama.chat(
        model="llama3.2-vision",
        messages=[{
            "role": "user",
            "content": "this is a frame of a video, just describe what is happening, be detailed about appearances and environment",
            "images": [temp_image_path]
        }],
    )

    # Extract the model's response
    cleaned_text = response['message']['content'].strip()
    os.remove(temp_image_path)  # Clean up the temporary file
    return jsonify({'description': cleaned_text})


@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        frames = data.get('descriptions', [])
        if not frames:
            return jsonify({'error': 'No descriptions provided'}), 400
        
        # Combine all frame descriptions into a single prompt
        frames_text = '\n'.join(frames)
        prompt = f"You are writing a police report from a video. Here are descriptions of each frame from a video:\n{frames_text}\n\nGenerate a complete and coherent description of the entire video, capturing the flow and context of the scene."
        
        # Generate the final report
        response = ollama.chat(
            model="llama2:7b",
            messages=[{"role": "system", "content": prompt}]
        )
        final_description = response['message']['content']
        return jsonify({'report': final_description}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to generate report'}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(port=4000)