from flask import Flask, request, jsonify
import urllib.request
from skimage import io, exposure, filters
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    urllib.request.urlretrieve(url, "temp.jpg")
    img = io.imread("temp.jpg")

    img_equalized = exposure.equalize_hist(img)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    img_unsharp = filters.unsharp_mask(img, radius=3, amount=2)

    # Convert float to uint8
    img_equalized = (img_equalized * 255).astype(np.uint8)
    img_adapteq = (img_adapteq * 255).astype(np.uint8)
    img_unsharp = (img_unsharp * 255).astype(np.uint8)

    # Metrics
    psnr_o = peak_signal_noise_ratio(img, img)
    ssim_o = structural_similarity(img, img, channel_axis=-1)

    psnr_eq = peak_signal_noise_ratio(img, img_equalized)
    ssim_eq = structural_similarity(img, img_equalized, channel_axis=-1)

    psnr_ad = peak_signal_noise_ratio(img, img_adapteq)
    ssim_ad = structural_similarity(img, img_adapteq, channel_axis=-1)

    psnr_us = peak_signal_noise_ratio(img, img_unsharp)
    ssim_us = structural_similarity(img, img_unsharp, channel_axis=-1)

    return jsonify({
        "Original": {"PSNR": psnr_o, "SSIM": ssim_o},
        "Histogram Equalization": {"PSNR": psnr_eq, "SSIM": ssim_eq},
        "Adaptive Equalization": {"PSNR": psnr_ad, "SSIM": ssim_ad},
        "Unsharp Masking": {"PSNR": psnr_us, "SSIM": ssim_us}
    })

if __name__ == '__main__':
    app.run(debug=True)
