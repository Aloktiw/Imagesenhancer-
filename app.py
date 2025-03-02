from flask import Flask, request, render_template, send_file
import numpy as np
import cv2
from io import BytesIO
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = Flask(__name__)

# Load AI model once at startup
model = RRDBNet(num_in_ch=3, num_out_ch=3)
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=400,  # Optimized for mobile processing
    tile_pad=10,
    pre_pad=0,
    half=False)  # Keep float32 for Android compatibility

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    # Mobile-optimized image processing
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # AI Enhancement (4x upscale)
    output, _ = upsampler.enhance(img, outscale=4)
    
    # Convert back to downloadable format
    _, img_encoded = cv2.imencode('.png', output)
    return send_file(
        BytesIO(img_encoded.tobytes()),
        mimetype='image/png',
        as_attachment=True,
        download_name='enhanced_image.png')

if __name__ == '__main__':
    app.run()