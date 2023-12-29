import os
from flask import Flask, render_template, jsonify, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from numpy.random import randn
from matplotlib import pyplot
import base64
from io import BytesIO

app = Flask(__name__)

GENERATED_FOLDER = 'static/generated'

app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# Load your GAN model from the H5 file
model = load_model('gan.h5')

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

def generate_images(model, latent_points):
    generated_images = model.predict(latent_points)
    return generated_images

# create a plot of generated images
def plot_generated(examples, n, image_size=(80, 80)): 
    # plot images
    fig, axes = pyplot.subplots(n, n, figsize=(8, 8))
    for i in range(n * n):
        # turn off axis
        axes.flatten()[i].axis('off')
        # plot raw pixel data
        axes.flatten()[i].imshow(examples[i, :, :])
    
    # Save the plot to BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pyplot.close(fig)
    
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/generate', methods=['POST'])
def generate():
    latent_dim = 100
    n_samples = 4
    latent_points = generate_latent_points(latent_dim, n_samples)
    generated_images = generate_images(model, latent_points)
    generated_images = (generated_images + 1) / 2.0
    img_data = plot_generated(generated_images, int(np.sqrt(n_samples)))

    return jsonify({'success': True, 'generated_image': img_data})

if __name__ == '__main__':
    app.run(debug=True)
