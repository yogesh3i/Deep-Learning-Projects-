import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps

# Import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

app = Flask(__name__)

# Constants
MODEL_PATH = 'digit_model.h5'
model = None
activation_model = None

# Load the trained CNN model
def load_deep_learning_model():
    global model, activation_model
    if not TENSORFLOW_AVAILABLE:
        print("[-] TensorFlow is not installed. Running in Demo Mode.")
        return
    
    # Auto-train the model if it is missing (highly useful for cloud deployments!)
    if not os.path.exists(MODEL_PATH):
        print("[*] Model file not found. Triggering automated training script (train.py)...")
        try:
            import subprocess
            # Run train.py to train the CNN and generate all assets
            subprocess.run(["python", "train.py"], check=True)
            print("[+] Automated training completed successfully!")
        except Exception as e:
            print(f"[-] Automated training failed: {e}. Falling back to Demo Mode.")
            
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[+] Loaded TensorFlow CNN model from {MODEL_PATH}")
            
            # Create activation model to visualize first and second Conv2D layer outputs
            conv_layers = []
            for layer in model.layers:
                if 'conv' in layer.name.lower():
                    conv_layers.append(layer.output)
            
            if len(conv_layers) >= 2:
                activation_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[conv_layers[0], conv_layers[1]]
                )
                print("[+] Successfully initialized CNN activation visualizer model.")
            else:
                print("[-] Could not find sufficient Conv2D layers for activation visualization.")
        except Exception as e:
            print(f"[-] Error loading model: {e}. Running in Demo Mode.")
    else:
        print(f"[-] Model file {MODEL_PATH} not found. Running in Demo Mode.")

# Initialise model loading
load_deep_learning_model()

def preprocess_image(base64_data):
    """
    Decodes a base64 image, preprocesses it to 28x28 grayscale, 
    inverts background to match MNIST (white digit, black background),
    and reshapes for model input.
    """
    # Remove header if present (e.g. 'data:image/png;base64,')
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
        
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))
    
    # Flatten transparent/alpha layers properly onto a solid black background
    if image.mode == 'RGBA':
        # Create solid black canvas of identical size
        background = Image.new('RGB', image.size, (0, 0, 0))
        # Paste drawing onto it using its alpha channel as a mask
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert('RGB')
        
    # Convert image to grayscale (single-channel intensity values)
    image = image.convert('L')
    
    # MNIST requires white strokes on black background
    # If user drew dark ink on white canvas, invert it
    avg_pixel = np.mean(np.array(image))
    if avg_pixel > 127:
        image = ImageOps.invert(image)
            
    # Resize to 28x28 pixels using Lanczos resampling
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to array, normalize to [0, 1]
    img_array = np.array(image).astype('float32') / 255.0
    
    # Reshape to (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def generate_activation_grid(feature_maps, grid_size=(4, 4), block_size=(40, 40)):
    """
    Takes a 3D feature map tensor (height, width, channels) and creates a single 
    base64-encoded composite grid image showing the first channels/filters.
    """
    # feature_maps shape: (H, W, C)
    h, w, c = feature_maps.shape
    rows, cols = grid_size
    num_filters = min(rows * cols, c)
    
    # Create blank canvas for the grid
    grid_w = cols * block_size[0]
    grid_h = rows * block_size[1]
    grid_img = Image.new('L', (grid_w, grid_h), color=0)
    
    for i in range(num_filters):
        f_map = feature_maps[:, :, i]
        
        # Normalize filter output to [0, 255]
        f_min, f_max = f_map.min(), f_map.max()
        if f_max > f_min:
            f_map = (f_map - f_min) / (f_max - f_min) * 255.0
        else:
            f_map = np.zeros_like(f_map)
            
        # Convert to PIL and resize to make it sharp and visible
        f_img = Image.fromarray(f_map.astype('uint8'))
        f_img = f_img.resize(block_size, Image.Resampling.NEAREST)
        
        # Paste into grid
        r = i // cols
        c_idx = i % cols
        grid_img.paste(f_img, (c_idx * block_size[0], r * block_size[1]))
        
    # Save composite to bytes and convert to base64
    buffered = io.BytesIO()
    grid_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def home():
    # If model is loaded, we are active, otherwise we are in demo mode
    status = "Active" if model is not None else "Demo Mode (Model Untrained)"
    return render_template('index.html', model_status=status)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Reload model if it was trained in the background since server started
    global model, activation_model
    if model is None:
        load_deep_learning_model()

    try:
        # Preprocess canvas image
        preprocessed_img = preprocess_image(data['image'])
        
        # Check if we are running in real model mode or demo fallback
        if model is not None:
            # 1. Run actual inference
            predictions = model.predict(preprocessed_img, verbose=0)[0]
            predicted_digit = int(np.argmax(predictions))
            confidences = [float(c) for c in predictions]
            
            # 2. Extract internal layer activations
            act_base64_1 = ""
            act_base64_2 = ""
            if activation_model is not None:
                activations = activation_model.predict(preprocessed_img, verbose=0)
                # activations[0] -> Layer 1 output: shape (1, 26, 26, 32)
                # activations[1] -> Layer 2 output: shape (1, 11, 11, 64)
                
                act_layer_1 = activations[0][0] # (26, 26, 32)
                act_layer_2 = activations[1][0] # (11, 11, 64)
                
                # Generate base64 grid images
                act_base64_1 = generate_activation_grid(act_layer_1, grid_size=(4, 4))
                act_base64_2 = generate_activation_grid(act_layer_2, grid_size=(4, 4))
            
            return jsonify({
                'mode': 'Deep Learning Model',
                'prediction': predicted_digit,
                'confidences': confidences,
                'activation1': act_base64_1,
                'activation2': act_base64_2
            })
            
        else:
            # Fallback Demo Mode if model is not trained yet
            # Generate simulated predictions based on drawing density or simple random distribution
            simulated_probs = np.random.dirichlet(np.ones(10) * 0.5) # highly skewed simulated probs
            predicted_digit = int(np.argmax(simulated_probs))
            confidences = [float(c) for c in simulated_probs]
            
            # Generate static simulated visual grids representing mock activation patterns
            mock_fmap_1 = np.random.rand(26, 26, 16)
            mock_fmap_2 = np.random.rand(11, 11, 16)
            
            act_base64_1 = generate_activation_grid(mock_fmap_1, grid_size=(4, 4))
            act_base64_2 = generate_activation_grid(mock_fmap_2, grid_size=(4, 4))
            
            return jsonify({
                'mode': 'Simulated Demo Mode (Please run python train.py first)',
                'prediction': predicted_digit,
                'confidences': confidences,
                'activation1': act_base64_1,
                'activation2': act_base64_2,
                'warning': 'Model not trained yet. Showing simulated predictions.'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Hugging Face Spaces binds to 0.0.0.0 and uses port 7860 (or passes a dynamic PORT env variable)
    port = int(os.environ.get('PORT', 7860))
    # We set debug=False in production mode
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
