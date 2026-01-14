import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, colorchooser, ttk, messagebox
from PIL import Image, ImageTk
import base64
import io
import json
import time
import requests 
import os 
import random # NEW: For random name generation

# --- GLOBAL VARIABLES & CONFIGURATION ---

# 1. Image Paths
DEFAULT_IMAGE_PATH = 'sample_image.jpg' 

# 2. Tkinter/OpenCV Image Holders
root = None
comparison_bgr_image = None  # Holds the second image for comparison
display_photo_images = {}    # Dictionary to hold Tkinter PhotoImage objects for all loaded files
loaded_files_list = []       # NEW: List to hold all loaded BGR arrays and file info: [{'name': str, 'bgr': array, 'display_id': int}]
image_grid_frame = None      # Frame to hold the dynamic grid of images

# --- NEW: RANDOM NAME GENERATOR ---
ADJECTIVES = [
    "azure", "silent", "brave", "calm", "digital", "electric", "fierce", "gentle", 
    "hidden", "icy", "jolly", "keen", "lucky", "misty", "neon", "odd", "proud", 
    "quiet", "rapid", "solar", "turbo", "urban", "vivid", "wild", "xenon", "young", "zealous"
]
NOUNS = [
    "apple", "bridge", "cloud", "dragon", "eagle", "forest", "ghost", "harbor", 
    "island", "jungle", "kite", "lion", "moon", "nebula", "ocean", "pixel", "quest", 
    "robot", "star", "tiger", "unicorn", "valley", "wolf", "xenon", "yacht", "zebra"
]

def get_random_filename():
    """Generates a random filename using an adjective and a noun."""
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    rand_id = random.randint(10, 99) 
    return f"generated_{adj}_{noun}_{rand_id}.png"

# !!! IMPORTANT: For local use, paste your Gemini API key here to avoid the 403 error !!!
# Load API key from config.json
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        api_key = config.get("gemini_api_key")
except FileNotFoundError:
    api_key = ""
    print("Warning: config.json not found.")

# --- FILTER VARIABLES ---
var_grayscale = None        
var_median_blur = None      
var_bilateral = None        
var_equalize_hist = None    
var_erode = None            
var_dilate = None           
var_invert = None           
var_threshold = None        
var_sepia = None            
var_edge_mode = None        
var_box_blur = None         
var_flip_h = None           
var_flip_v = None           
var_channel_swap = None     
var_sharpen = None          
var_custom_tint_toggle = None
custom_tint_bgr = np.array([0, 0, 0], dtype=np.uint8) # Default BGR (Black)
var_cartoon_filter = None # NEW
var_posterization_filter = None # NEW

# --- SLIDER WIDGET PLACEHOLDERS (Must be declared before update_preview_grid uses them) ---
slider_brightness = None
slider_contrast = None
slider_hue = None
slider_saturation = None
slider_value = None
slider_scale_factor = None
slider_gaussian_blur = None
slider_gamma = None
slider_pixelation_block_size = None
slider_poster_colors = None # NEW

# --- AI VARIABLES ---
ai_prompt_entry_reimagine = None # For Image-to-Image tab
ai_prompt_entry_generate = None  # For Text-to-Image tab
ai_num_generations = None        # New: Entry for number of generations
ai_loading_label = None          # Status label placeholder (Defined later in GUI Setup)
ai_reimagine_button = None 
ai_generate_button = None
is_ai_generating = False # Flag to prevent concurrent generation

# --- STYLE PRESETS ---
STYLE_PRESETS = {
    "None": "",
    "Studio Ghibli": "in the whimsical, hand-painted style of Studio Ghibli, lush backgrounds, soft lighting",
    "Makoto Shinkai": "hyper-detailed anime style, vibrant lighting, cinematic atmosphere, breathtaking skies",
    "Cyberpunk Noir": "cyberpunk aesthetic, neon lights, rainy night, high contrast, cinematic noir",
    "Spider-Verse": "stylized comic book style, halftone patterns, vibrant colors, expressive ink lines",
    "Ukiyo-e": "traditional Japanese woodblock print style, flat colors, elegant linework",
    "Watercolor": "soft watercolor painting, bleeding colors, textured paper, artistic and hand-drawn",
    "Lo-fi Aesthetic": "lo-fi hip hop aesthetic, muted retro colors, cozy atmosphere, slight grain",
    "Golden Hour": "lit by the warm glow of the setting sun, long shadows, volumetric lighting, nostalgic"
}


# --- HELPER FUNCTIONS FOR AI ---

def numpy_to_base64(numpy_array):
    """Converts a BGR NumPy array to a base64 encoded PNG string."""
    # Ensure image is RGB (standard for API input)
    # NOTE: This conversion is correct for preparing OpenCV (BGR) data for the API (RGB expectation)
    # rgb_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB) # <-- REMOVED THIS LINE TO FIX BLUE SKIN BUG
    
    # Encode as PNG data
    is_success, buffer = cv2.imencode(".png", numpy_array)
    if is_success:
        return base64.b64encode(buffer).decode("utf-8")
    return None

def base64_to_numpy(base64_string):
    """Converts a base64 encoded PNG string back to a BGR NumPy array (without color correction)."""
    try:
        # Decode base64 to raw bytes
        img_bytes = base64.b64decode(base64_string)
        # Create a NumPy array from the bytes
        img_array = np.frombuffer(img_bytes, np.uint8)
        # Decode the image data. This returns data in the order OpenCV expects (BGR) 
        # but the actual channels might be swapped if the source bytes were RGB.
        decoded_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if decoded_array is not None:
            # Removed global RGB->BGR swap. 
            # The swap must now be applied locally in the AI action functions if needed.
            return decoded_array 
        
        return None
    except Exception as e:
        print(f"Error decoding base64 to numpy: {e}")
        return None

def start_ai_cooldown(seconds=15):
    """Disables the AI buttons and starts a countdown timer."""
    global ai_reimagine_button, ai_generate_button, ai_loading_label
    
    # Disable both buttons
    if ai_reimagine_button:
        ai_reimagine_button.config(state=tk.DISABLED)
    if ai_generate_button:
        ai_generate_button.config(state=tk.DISABLED)
        
    ai_loading_label.config(fg='red')
    
    def countdown(remaining):
        if remaining > 0:
            ai_loading_label.config(text=f"Rate Limit Cooldown: {remaining}s")
            root.after(1000, countdown, remaining - 1)
        else:
            # Re-enable buttons
            if ai_reimagine_button:
                ai_reimagine_button.config(state=tk.NORMAL)
            if ai_generate_button:
                ai_generate_button.config(state=tk.NORMAL)
            ai_loading_label.config(text="Ready.", fg='black')

    countdown(seconds)

# --- GEMINI IMAGE-TO-IMAGE FUNCTION ---
def fetch_gemini_image(prompt, image_b64):
    """Makes a POST request to the Gemini API for image generation (single attempt with timeout)."""
    
    model_name = "gemini-2.5-flash-image-preview"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": image_b64}}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        
        # Check for 429 error specifically
        if response.status_code == 429:
             raise requests.exceptions.RequestException("429 Client Error: Too Many Requests (Rate Limit Hit)")

        result = response.json()
        base64_data = result['candidates'][0]['content']['parts'][0]['inlineData']['data']
        return base64_data

    except requests.exceptions.Timeout:
        raise Exception("API request timed out after 120 seconds. The model did not respond in time.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to generate image from AI: {e}")
    except Exception as e:
        raise Exception(f"Failed to process AI response: {e}")
    
# --- IMAGEN TEXT-TO-IMAGE FUNCTION ---
def fetch_imagen_image(prompt):
    """Makes a POST request to the Imagen API for text-to-image generation."""

    model_name = "imagen-4.0-generate-001"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict?key={api_key}"

    payload = {
        "instances": {"prompt": prompt},
        "parameters": {"sampleCount": 1}
    }

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()

        # Check for 429 error specifically
        if response.status_code == 429:
             raise requests.exceptions.RequestException("429 Client Error: Too Many Requests (Rate Limit Hit)")
             
        result = response.json()
        base64_data = result['predictions'][0]['bytesBase64Encoded']
        return base64_data

    except requests.exceptions.Timeout:
        raise Exception("API request timed out after 120 seconds. The model did not respond in time.")
    except requests.exceptions.RequestException as e:
        # Check for 403 Forbidden (Missing Key/Permissions) or 429 (Rate Limit)
        if "429 Client Error" in str(e): # Already handled by specific check above, but keep generic handler
             raise Exception(f"AI Quota Exceeded. Please wait for the daily/minute quota to reset.")
        if response.status_code == 403:
             raise Exception(f"Forbidden (403). Check API key and billing status.")
        raise Exception(f"Failed to generate image from AI: {e}")
    except Exception as e:
        raise Exception(f"Failed to process AI response: {e}")

# --- AI ACTION FUNCTIONS ---

def generate_new_image_action():
    """Triggers the Text-to-Image generation process and loads the new image(s)."""
    global loaded_files_list, ai_loading_label, is_ai_generating
    
    if is_ai_generating:
        return 
        
    prompt = ai_prompt_entry_generate.get("1.0", tk.END).strip()
    if not prompt:
        tk.messagebox.showerror("Error", "Please enter a prompt for image generation.")
        return

    try:
        num_generations = int(ai_num_generations.get())
        if not 1 <= num_generations <= 10:
             tk.messagebox.showerror("Error", "Number of generations must be between 1 and 10.")
             return
    except ValueError:
        tk.messagebox.showerror("Error", "Invalid number of generations.")
        return
        
    ai_loading_label.config(text=f"Generating 1 of {num_generations}... Please wait.", fg='blue')
    is_ai_generating = True
    root.update()
    
    successful_generations = 0
    loaded_files_list.clear() # Clear existing list for new batch

    # Need a mutable list copy to handle retries cleanly without infinite loops
    generations_to_process = list(range(num_generations))
    
    while generations_to_process:
        # i is the original 1-based index (for file naming/display purposes)
        i = generations_to_process.pop(0) + 1 
        
        try:
            # We use successful_generations + 1 for the current attempt number in the UI
            ai_loading_label.config(text=f"Generating {successful_generations + 1} of {num_generations}...", fg='blue')
            root.update()
            
            # 1. Call the API for text-to-image generation
            generated_b64 = fetch_imagen_image(prompt)
            
            # 2. Decode the result
            generated_array = base64_to_numpy(generated_b64)
            
            if generated_array is not None:
                #generated_array = cv2.cvtColor(generated_array, cv2.COLOR_RGB2BGR)
                # 3. Add the new image
                # --- CHANGE: USE RANDOM NAME GENERATOR ---
                new_file_name = get_random_filename()
                loaded_files_list.append({'name': new_file_name, 'bgr': generated_array})
                successful_generations += 1
            else:
                tk.messagebox.showerror("Error", f"Generation {i} failed to process image data.")

        except Exception as e:
            if "429 Client Error" in str(e):
                tk.messagebox.showwarning("Rate Limit", f"Rate limit hit at generation {i}. Waiting 15 seconds to resume...")
                start_ai_cooldown(15)
                time.sleep(15)
                # Re-insert the failed item index to retry
                generations_to_process.insert(0, i - 1) 
                # After sleeping, we loop back to continue.
            else:
                tk.messagebox.showerror("AI Error", f"Fatal error on generation {i}: {e}")
                break
        
    if successful_generations > 0:
        tk.messagebox.showinfo("Batch Complete", f"Successfully generated and loaded {successful_generations} new image(s)!")
        update_preview_grid()
    else:
        tk.messagebox.showwarning("Generation Failed", "No images were successfully generated or loaded.")


    # Reset state and start final cooldown
    ai_loading_label.config(text="Ready.", fg='black')
    is_ai_generating = False
    start_ai_cooldown(15) 


def reimagine_image_action():
    """Triggers the Gemini image-to-image generation process."""
    global loaded_files_list, ai_loading_label, is_ai_generating
    
    if is_ai_generating or not loaded_files_list:
        return 

    # 1. Get User Text + Style Tag
    raw_prompt = ai_prompt_entry_reimagine.get("1.0", tk.END).strip()
    selected_style = var_style_selection.get()
    style_tag = STYLE_PRESETS.get(selected_style, "")
    
    # "Sandwich" the prompt
    full_prompt = f"{raw_prompt}, {style_tag}" if style_tag else raw_prompt

    is_ai_generating = True
    root.update()
    
    # Process the first image in the grid
    file_info = loaded_files_list[0]
    source_bgr = file_info['bgr']
    source_name = file_info['name']

    ai_loading_label.config(text=f"Reimaging: {source_name} as {selected_style}...", fg='blue')
    root.update()

    try:
        # 2. Encode
        image_b64 = numpy_to_base64(source_bgr)
        
        # 3. API Call
        generated_b64 = fetch_gemini_image(full_prompt, image_b64)
        
        # 4. Decode (Base64 -> BGR NumPy)
        generated_array = base64_to_numpy(generated_b64)

        if generated_array is not None:
            # We NO LONGER need cvtColor here if base64_to_numpy uses imdecode
            # --- CHANGE: USE RANDOM NAME GENERATOR ---
            new_file_name = get_random_filename()
            loaded_files_list.append({'name': new_file_name, 'bgr': generated_array})
            
            update_preview_grid()
            tk.messagebox.showinfo("Success", f"Image reimagined in {selected_style} style!")
        else:
            tk.messagebox.showerror("Error", "Could not process AI response.")

    except Exception as e:
        tk.messagebox.showerror("AI Error", f"Fatal error: {e}")

    ai_loading_label.config(text="Ready.", fg='black')
    is_ai_generating = False
    start_ai_cooldown(15)


# --- IMAGE PROCESSING LOGIC (CV) ---

def process_image(bgr_array, brightness, contrast, hue, saturation, value, grayscale_mode, blur_ksize, scale_factor, gamma_factor, pixel_block_size):
    """Applies all slider and button effects sequentially to the provided image array."""
    
    if bgr_array is None:
        return None
    
    # 1. Start with a copy
    processed_image = bgr_array.copy()
    h_orig, w_orig = processed_image.shape[:2]

    # --- F. Scale / Resize ---
    if scale_factor != 100:
        scale = scale_factor / 100.0
        new_width = int(w_orig * scale)
        new_height = int(h_orig * scale)
        
        # TOGGLE LOGIC:
        # If High Quality is ON and we are shrinking, use AREA.
        # Otherwise, use standard CUBIC.
        if scale < 1.0 and var_high_quality_downscale.get():
            interpolation_method = cv2.INTER_AREA
        else:
            interpolation_method = cv2.INTER_CUBIC
        
        processed_image = cv2.resize(processed_image, 
                                     (new_width, new_height), 
                                     interpolation=interpolation_method)
        
    # --- NEW: Square (Letterbox) Padding ---
    # We apply this AFTER scaling so the padding matches the new size
    if var_square_padding.get():
        h, w = processed_image.shape[:2]
        max_dim = max(h, w)
        
        # Create a black canvas (zeros) of the max dimension
        # (If you want white padding, multiply by 255)
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # Calculate centering offsets
        x_off = (max_dim - w) // 2
        y_off = (max_dim - h) // 2
        
        # Paste the image into the center
        square_img[y_off:y_off+h, x_off:x_off+w] = processed_image
        processed_image = square_img
    
    # --- Image Flipping (Geometry) ---
    if var_flip_h.get():
        processed_image = cv2.flip(processed_image, 1) # Flip Horizontally
    if var_flip_v.get():
        processed_image = cv2.flip(processed_image, 0) # Flip Vertically

    # --- Pixelation Filter ---
    if pixel_block_size > 1:
        h, w = processed_image.shape[:2]
        block = int(pixel_block_size)
        
        small_w = max(1, int(w / block))
        small_h = max(1, int(h / block))
        
        temp_image = cv2.resize(processed_image, (small_w, small_h), 
                                interpolation=cv2.INTER_NEAREST)
        
        processed_image = cv2.resize(temp_image, (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
    
    # --- Median Blur Toggle (Denoise) ---
    if var_median_blur.get():
        processed_image = cv2.medianBlur(processed_image, 5)

    # --- Bilateral Filter (Edge-Preserving Denoising) ---
    if var_bilateral.get():
        processed_image = cv2.bilateralFilter(processed_image, 9, 75, 75)
    
    # --- Color Quantization (Posterization) ---
    if var_posterization_filter.get():
        K = slider_poster_colors.get()
        if K < 2: K = 2
        
        # Reshape the image to be a list of pixels (H*W, 3)
        Z = processed_image.reshape((-1, 3))
        Z = np.float32(Z)
        
        # Define criteria for K-Means (max 10 iterations, epsilon 1.0)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Apply K-Means
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to 8 bit and reshape to original image size
        center = np.uint8(center)
        res = center[label.flatten()]
        processed_image = res.reshape((processed_image.shape))


    # --- Morphological Operations (Erosion and Dilation) ---
    if var_erode.get() or var_dilate.get():
        kernel = np.ones((3, 3), np.uint8)
        if var_erode.get():
            processed_image = cv2.erode(processed_image, kernel, iterations=1)
        if var_dilate.get():
            processed_image = cv2.dilate(processed_image, kernel, iterations=1)
            
    # --- Sharpening Filter ---
    if var_sharpen.get():
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]], dtype=np.float32)
        processed_image = cv2.filter2D(processed_image, -1, sharpen_kernel)

    # --- Sepia Filter (Color Mapping via Matrix) ---
    if var_sepia.get():
        sepia_kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ], dtype=np.float32)
        processed_image_float = processed_image.astype(np.float32)
        processed_image = cv2.transform(processed_image_float, sepia_kernel)
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

    # --- Color Channel Swap (BGR <-> RGB Filter Effect) ---
    if var_channel_swap.get():
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # --- A. Brightness and Contrast (Alpha/Beta adjustment) ---
    alpha = contrast / 100.0  
    beta = brightness - 100   
    processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
    
    # --- Gamma Correction (Non-linear Brightness Adjustment) ---
    if gamma_factor != 100:
        gamma = gamma_factor / 100.0
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        processed_image = cv2.LUT(processed_image, table)
    
    # --- Histogram Equalization (Contrast Enhancement) ---
    if var_equalize_hist.get():
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_equalized = cv2.equalizeHist(v)
        processed_image = cv2.merge([h, s, v_equalized])
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)
        
    # --- B. HSV Manipulation (Hue, Saturation, Value) ---
    if hue != 100 or saturation != 100 or value != 100:
        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        s_factor = saturation / 100.0
        v_factor = value / 100.0
        s = np.clip(s * s_factor, 0, 255).astype(np.uint8)
        v = np.clip(v * v_factor, 0, 255).astype(np.uint8)
        
        hue_shift = hue - 100
        h = np.clip(h.astype(int) + hue_shift, 0, 180).astype(np.uint8) 
        
        processed_image = cv2.merge([h, s, v])
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)
    
    # --- Custom Color Tint (HUE/SATURATION BLEND) ---
    if var_custom_tint_toggle.get():
        # 1. Convert image to HSV
        hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        h_img, s_img, v_img = cv2.split(hsv_image)
        
        # 2. Convert selected BGR color to HSV to get the target H and S
        color_bgr_array = np.uint8([[custom_tint_bgr]])
        color_hsv = cv2.cvtColor(color_bgr_array, cv2.COLOR_BGR2HSV)[0, 0]
        
        h_new = color_hsv[0] # Target Hue (0-179)
        s_new = color_hsv[1] # Target Saturation (0-255)
        
        # 3. Create new H and S channels the size of the image, filled with target values
        h_new_channel = np.full_like(h_img, h_new, dtype=np.uint8)
        s_new_channel = np.full_like(s_img, s_new, dtype=np.uint8)
        
        # 4. Merge the target H and S channels with the image's original Value (V) channel
        blended_hsv = cv2.merge([h_new_channel, s_new_channel, v_img])
        
        # 5. Convert back to BGR
        processed_image = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2BGR)


    # --- Box Blur Toggle ---
    if var_box_blur.get():
        processed_image = cv2.blur(processed_image, (5, 5))

    # --- C. Gaussian Blur ---
    k = int(slider_gaussian_blur.get()) # Use the global slider var
    if k > 0 and k % 2 == 0:
        k += 1 
    
    if k > 0:
        processed_image = cv2.GaussianBlur(processed_image, (k, k), 0)

    # --- Cartoon/Stylization Filter ---
    if var_cartoon_filter.get():
        # Uses a non-photorealistic rendering filter built into OpenCV
        processed_image = cv2.stylization(processed_image, sigma_s=60, sigma_r=0.45)

    # --- Image Inversion (Negative Filter) ---
    if var_invert.get():
        processed_image = cv2.bitwise_not(processed_image)

    # --- Binary Thresholding ---
    if var_threshold.get():
        gray_image_for_thresh = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        _, processed_image_thresh = cv2.threshold(gray_image_for_thresh, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.cvtColor(processed_image_thresh, cv2.COLOR_GRAY2BGR)


    # --- D. Edge Detection / Laplacian / Sobel ---
    edge_mode = var_edge_mode.get()
    
    if edge_mode != "None":
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        if edge_mode == "Canny":
            processed_image = cv2.Canny(gray_image, 100, 200)
        
        elif edge_mode == "Laplacian":
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            processed_image = cv2.convertScaleAbs(laplacian)
            
        elif edge_mode.startswith("Sobel"):
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

            if edge_mode == "SobelX":
                processed_image = cv2.convertScaleAbs(sobelx)
            elif edge_mode == "SobelY":
                processed_image = cv2.convertScaleAbs(sobely)
            elif edge_mode == "SobelComplete":
                abs_sobelx = cv2.convertScaleAbs(sobelx)
                abs_sobely = cv2.convertScaleAbs(sobely)
                processed_image = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
        
        return cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)


    # --- E. Grayscale Toggle (Only runs if no edge detection is active) ---
    if grayscale_mode:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    return processed_image

# --- NEW: COMPARISON LOGIC ---

def calculate_mse(imageA, imageB):
    """Calculates the Mean Squared Error (MSE) between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err

def load_compare_image():
    """Opens a file dialog to select the second image for comparison."""
    global comparison_bgr_image
    file_path = filedialog.askopenfilename(
        title="Select Second Image for Comparison",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is not None:
            comparison_bgr_image = img
            tk.messagebox.showinfo("Image Loaded", "Comparison image loaded successfully. Click 'Calculate MSE' to compare.")
        else:
            tk.messagebox.showerror("Error", "Could not load the comparison image.")

def compare_action():
    """Performs the image comparison and displays the MSE result."""
    global loaded_files_list, comparison_bgr_image
    
    if not loaded_files_list:
        tk.messagebox.showerror("Error", "Please load the primary image first.")
        return
    if comparison_bgr_image is None:
        tk.messagebox.showerror("Error", "Please load the comparison image first.")
        return

    # Use the first loaded image as the primary image for comparison
    primary_bgr_image = loaded_files_list[0]['bgr']
    
    # 1. Ensure the comparison image is resized to match the primary image dimensions
    height, width, _ = primary_bgr_image.shape
    resized_compare_image = cv2.resize(comparison_bgr_image, (width, height), 
                                       interpolation=cv2.INTER_CUBIC)

    # 2. Get the processed version of the primary image (applying all current filters)
    brightness = slider_brightness.get()
    contrast = slider_contrast.get()
    hue = slider_hue.get()
    saturation = slider_saturation.get()
    value = slider_value.get()
    grayscale_mode = var_grayscale.get()
    blur_ksize = slider_gaussian_blur.get()
    scale_factor = slider_scale_factor.get()
    gamma_factor = slider_gamma.get()
    pixel_block_size = slider_pixelation_block_size.get()

    processed_original_image = process_image(primary_bgr_image, brightness, contrast, hue, saturation, value, grayscale_mode, blur_ksize, scale_factor, gamma_factor, pixel_block_size)

    # 3. Calculate MSE
    mse = calculate_mse(processed_original_image, resized_compare_image)
    
    # Provide context for the result
    if mse < 100:
        result_msg = "Extremely Similar (MSE < 100)"
    elif mse < 500:
        result_msg = "Highly Similar (MSE < 500)"
    elif mse < 1500:
        result_msg = "Moderately Different"
    else:
        result_msg = "Significantly Different"
    
    info_msg = (f"Comparison Result: {result_msg}\n"
                f"Mean Squared Error (MSE): {mse:.2f}\n\n"
                f"Comparing processed '{loaded_files_list[0]['name']}' vs. loaded comparison image.")

    tk.messagebox.showinfo("Image Comparison Complete", info_msg)

# --- NEW: DELETE ACTION ---
def delete_image_action(index):
    """Removes the image at the specified index and refreshes the grid."""
    global loaded_files_list
    if 0 <= index < len(loaded_files_list):
        del loaded_files_list[index]
        update_preview_grid()


def update_preview_grid(*args):
    """
    Clears the image grid and redraws all loaded images with the current filter settings.
    """
    global root, loaded_files_list, display_photo_images, image_grid_frame
    
    # Clear the existing grid content
    for widget in image_grid_frame.winfo_children():
        widget.destroy()
    
    if not loaded_files_list:
        tk.Label(image_grid_frame, text="Load images to begin processing.").pack(expand=True, padx=50, pady=50)
        return

    # 1. Read values from GUI controls
    # Check which tab is currently selected
    selected_tab = notebook.index(notebook.select())

    # Safely read slider values using .get(), relying on global assignment during create_slider
    if selected_tab == 0:
        brightness = slider_brightness.get()
        contrast = slider_contrast.get()
        hue = slider_hue.get()
        saturation = slider_saturation.get()
        value = slider_value.get()
        grayscale_mode = var_grayscale.get()
        blur_ksize = slider_gaussian_blur.get()
        scale_factor = slider_scale_factor.get()
        gamma_factor = slider_gamma.get()
        pixel_block_size = slider_pixelation_block_size.get()
    else:
        # AI Manipulation Tab (AI tab doesn't need to read CV slider values)
        brightness, contrast, hue, saturation, value = 100, 100, 100, 100, 100
        grayscale_mode = False
        blur_ksize, scale_factor, gamma_factor, pixel_block_size = 1, 100, 100, 1
    
    # Grid layout parameters
    cols = 2
    max_preview_size = 300 # Max dimension for any single preview
    
    # 2. Process and display each image
    for i, file_info in enumerate(loaded_files_list):
        bgr_array = file_info['bgr']
        
        # Process the image using the current slider/toggle settings
        processed_array = process_image(bgr_array, brightness, contrast, hue, saturation, value, grayscale_mode, blur_ksize, scale_factor, gamma_factor, pixel_block_size)
        
        if processed_array is not None:
            # Convert NumPy array (BGR) to PIL Image (RGB)
            img_rgb = cv2.cvtColor(processed_array, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Resize image to fit the preview size, maintaining aspect ratio
            width, height = pil_img.size
            if width > height:
                ratio = max_preview_size / width
                new_size = (max_preview_size, int(height * ratio))
            else:
                ratio = max_preview_size / height
                new_size = (int(width * ratio), max_preview_size)
            
            pil_img = pil_img.resize(new_size)
                
            # Convert PIL image to Tkinter PhotoImage
            photo_key = file_info['name']
            display_photo_images[photo_key] = ImageTk.PhotoImage(pil_img)

            # Create frame for image and label
            img_frame = tk.Frame(image_grid_frame, bd=2, relief=tk.RIDGE)
            
            # Image Label
            img_label = tk.Label(img_frame, image=display_photo_images[photo_key])
            img_label.pack()
            
            # File Name Label
            tk.Label(img_frame, text=file_info['name'], font=('Arial', 9)).pack()

            # --- NEW: Delete Button ---
            # Creates a small red 'x' button in the top right corner of the individual image frame
            btn_del = tk.Button(img_frame, text="x", bg="red", fg="white", 
                                font=("Arial", 8, "bold"), bd=0, padx=2, pady=0,
                                command=lambda idx=i: delete_image_action(idx))
            btn_del.place(relx=1.0, rely=0.0, anchor="ne")
            # --------------------------

            # Place frame in the grid
            row = i // cols
            col = i % cols
            img_frame.grid(row=row, column=col, padx=10, pady=10)
        
    # Update the scroll region of the grid frame
    image_grid_frame.update_idletasks()
    # The image_grid_frame is the window inside the main canvas, we need to update the main canvas's scroll region
    control_canvas_display.config(scrollregion=control_canvas_display.bbox("all"))

def load_image():
    """Opens a file dialog to select one or more new images."""
    global loaded_files_list
    file_paths = filedialog.askopenfilenames(
        title="Select Image File(s) (PNG, JPG, BMP)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if file_paths:
        # Clear previous list if it's the start of a new batch
        loaded_files_list.clear() 
        
        loaded_count = 0
        for file_path in file_paths:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                file_name = file_path.split('/')[-1] # Get just the file name
                loaded_files_list.append({'name': file_name, 'bgr': img})
                loaded_count += 1
            
        if loaded_count > 0:
            reset_sliders(trigger_update=False)
            update_preview_grid()
            tk.messagebox.showinfo("Success", f"Successfully loaded {loaded_count} image(s).")
        else:
            tk.messagebox.showerror("Error", "Could not load any selected images.")


def reset_sliders(trigger_update=True):
    """Resets all control sliders and checkboxes to their default (neutral) positions."""
    global comparison_bgr_image, custom_tint_bgr
    
    # Sliders
    slider_brightness.set(100)
    slider_contrast.set(100)
    slider_hue.set(100)
    slider_saturation.set(100)
    slider_value.set(100)
    slider_scale_factor.set(100)
    slider_gaussian_blur.set(1)
    slider_gamma.set(100)        
    slider_pixelation_block_size.set(1)
    slider_poster_colors.set(8) # NEW: Reset Poster Colors
    
    # Toggles
    var_grayscale.set(False)
    var_median_blur.set(False) 
    var_bilateral.set(False)    
    var_cartoon_filter.set(False) # NEW: Reset Cartoon Filter
    var_posterization_filter.set(False) # NEW: Reset Posterization Filter
    var_equalize_hist.set(False) 
    var_erode.set(False)          
    var_dilate.set(False)         
    var_invert.set(False)         
    var_threshold.set(False)      
    var_sepia.set(False)          
    var_box_blur.set(False)       
    var_flip_h.set(False)         
    var_flip_v.set(False)         
    var_channel_swap.set(False)   
    var_sharpen.set(False)        
    var_custom_tint_toggle.set(False) # Reset custom tint toggle
    var_edge_mode.set("None")    
    
    comparison_bgr_image = None  # Clear comparison image
    custom_tint_bgr = np.array([0, 0, 0], dtype=np.uint8) # Reset custom tint color to black
    
    if trigger_update:
        update_preview_grid()

def select_tint_color():
    """Opens a color chooser dialog and saves the selected color in BGR format."""
    global custom_tint_bgr
    # The askcolor dialog returns (RGB tuple, hex string)
    color_code = colorchooser.askcolor(title="Choose Custom Tint Color")
    
    if color_code and color_code[0]:
        r, g, b = color_code[0]
        # Store as BGR (OpenCV format)
        custom_tint_bgr = np.array([b, g, r], dtype=np.uint8)
        print(f"Custom tint color set to BGR: {custom_tint_bgr}")
        
        # Automatically enable the tint and update preview
        var_custom_tint_toggle.set(True)
        update_preview_grid()

# --- NEW SAVE LOGIC FUNCTION ---
def save_image_action(save_window, directory, save_vars, filename_vars, target_files):
    """Handles saving the images selected via the toggles in the save window."""
    
    saved_count = 0
    
    # Get current slider settings (as they apply to all processed images)
    # The fix ensures these slider objects are already defined globally
    brightness = slider_brightness.get()
    contrast = slider_contrast.get()
    hue = slider_hue.get()
    saturation = slider_saturation.get()
    value = slider_value.get()
    grayscale_mode = var_grayscale.get()
    blur_ksize = slider_gaussian_blur.get()
    scale_factor = slider_scale_factor.get()
    gamma_factor = slider_gamma.get()
    pixel_block_size = slider_pixelation_block_size.get() 
    
    
    for idx, file_info in enumerate(target_files):
        # Check if the toggle for this image is set (based on its index)
        if save_vars[idx].get():
            
            # Get the user-modified filename from the StringVar
            new_filename = filename_vars[idx].get()
            
            # 1. Process the image (using original BGR array + current filters)
            final_image_to_save = process_image(file_info['bgr'], brightness, contrast, hue, saturation, value, grayscale_mode, blur_ksize, scale_factor, gamma_factor, pixel_block_size)
            
            if final_image_to_save is not None:
                # 2. Construct the full save path using the new filename
                save_path = os.path.join(directory, new_filename)
                
                # 3. Save the file
                cv2.imwrite(save_path, final_image_to_save)
                saved_count += 1
                
    if saved_count > 0:
        tk.messagebox.showinfo("Save Complete", f"Successfully saved {saved_count} image(s) to:\n{directory}")
        save_window.destroy()
    else:
        tk.messagebox.showerror("Save Canceled", "No images were selected or saved.")


# --- MAIN SAVE ENTRY POINT (NEW) ---
def save_image():
    """Prompts for a save directory and opens the visual selection save window."""
    global loaded_files_list
    if not loaded_files_list:
        tk.messagebox.showerror("Error", "No image loaded to save.")
        return

    # --- CHANGE: HARDCODED SAVE PATH ---
    # Create an "edits" folder if it doesn't exist
    save_dir = os.path.join(os.getcwd(), "edits")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Determine which files DO NOT exist in the chosen directory
    files_to_save = []
    
    for file_info in loaded_files_list:
        file_path = os.path.join(save_dir, file_info['name'])
        
        # Only include the file if the filename does NOT exist in the directory
        if not os.path.exists(file_path):
            files_to_save.append(file_info)

    if not files_to_save:
        tk.messagebox.showinfo("Information", 
                               f"All loaded image filenames already exist in:\n{save_dir}\nNo images available to save without overwriting.")
        return

    # 3. Open the visual save selection window
    open_save_selection_window(save_dir, files_to_save)


def open_save_selection_window(save_dir, files_to_save):
    """Creates a new window allowing the user to visually select files to save."""
    
    save_window = tk.Toplevel(root)
    save_window.title(f"Select Images to Save ({save_dir})")
    
    tk.Label(save_window, text="Select processed images to save:", font=('Arial', 12, 'bold')).pack(pady=10)
    tk.Label(save_window, text=f"Destination: {save_dir}", fg='blue').pack(pady=5)

    # Setup scrollable area for previews
    save_canvas = tk.Canvas(save_window)
    save_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    save_scrollbar = tk.Scrollbar(save_window, orient=tk.VERTICAL, command=save_canvas.yview)
    save_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    save_canvas.configure(yscrollcommand=save_scrollbar.set)

    preview_frame = tk.Frame(save_canvas)
    save_canvas.create_window((0, 0), window=preview_frame, anchor="nw")

    def on_preview_frame_configure(event):
        save_canvas.configure(scrollregion=save_canvas.bbox("all"))
    preview_frame.bind("<Configure>", on_preview_frame_configure)
    
    
    # 4. Generate Previews and Toggles
    save_vars = [] # List to hold the BooleanVar for each toggle
    filename_vars = [] # NEW: List to hold the StringVar for each filename
    temp_photo_images = {} # Temporary dictionary to hold PhotoImages for this window

    # Get processed image settings
    brightness = slider_brightness.get()
    contrast = slider_contrast.get()
    hue = slider_hue.get()
    saturation = slider_saturation.get()
    value = slider_value.get()
    grayscale_mode = var_grayscale.get()
    blur_ksize = slider_gaussian_blur.get()
    scale_factor = slider_scale_factor.get()
    gamma_factor = slider_gamma.get()
    pixel_block_size = slider_pixelation_block_size.get()

    for i, file_info in enumerate(files_to_save):
        
        # A. Process the image (using original BGR array + current filters)
        processed_array = process_image(file_info['bgr'], brightness, contrast, hue, saturation, value, grayscale_mode, blur_ksize, scale_factor, gamma_factor, pixel_block_size)
        
        # B. Prepare the preview image
        img_rgb = cv2.cvtColor(processed_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((150, 150)) # Fixed size for save window preview
        
        photo_key = f"save_{file_info['name']}"
        temp_photo_images[photo_key] = ImageTk.PhotoImage(pil_img)
        
        # C. Create UI element
        img_frame = tk.Frame(preview_frame, bd=2, relief=tk.RIDGE, padx=5, pady=5)
        img_frame.grid(row=0, column=i, padx=10, pady=10)
        
        tk.Label(img_frame, image=temp_photo_images[photo_key]).pack()
        
        # D. Filename Entry (NEW)
        var_filename = tk.StringVar(value=file_info['name'])
        filename_vars.append(var_filename)
        tk.Entry(img_frame, textvariable=var_filename, width=20).pack(pady=2)
        
        # E. Create Toggle Button
        var_save = tk.BooleanVar(value=True) # Renamed var to var_save
        save_vars.append(var_save)
        
        tk.Checkbutton(img_frame, text="Save", variable=var_save, anchor='w').pack()

    # 5. Save Button (at the bottom)
    tk.Button(save_window, text="SAVE SELECTED IMAGES NOW", 
              command=lambda: save_image_action(save_window, save_dir, save_vars, filename_vars, files_to_save), 
              bg='green', fg='white', font=('Arial', 10, 'bold')).pack(pady=10)
    
    # Keep references to prevent garbage collection of image previews
    save_window.temp_photo_images = temp_photo_images


# --- GUI SETUP ---

# Create the main window
root = tk.Tk()
root.title("OpenCV Multi-Image Filter GUI")

# --- Control Panel (Left Side, made scrollable) ---
main_control_frame = tk.Frame(root, padx=5, pady=5)
main_control_frame.pack(side=tk.LEFT, fill=tk.Y)

# --- Notebook (Tabbed Interface) ---
notebook = ttk.Notebook(main_control_frame)
notebook.pack(fill='both', expand=True)
notebook.bind("<<NotebookTabChanged>>", update_preview_grid) # Re-render on tab switch


# --- Tab 1: Image Manipulation (CV) ---
tab_cv = tk.Frame(notebook)
notebook.add(tab_cv, text='Image Manipulation (CV)')

# Create scrollable frame setup for the CV tab contents
control_canvas_cv = tk.Canvas(tab_cv)
control_canvas_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control_scrollbar_cv = tk.Scrollbar(tab_cv, orient=tk.VERTICAL, command=control_canvas_cv.yview)
control_scrollbar_cv.pack(side=tk.RIGHT, fill=tk.Y)

control_canvas_cv.configure(yscrollcommand=control_scrollbar_cv.set)

inner_control_frame_cv = tk.Frame(control_canvas_cv, padx=5, pady=5)
control_canvas_cv.create_window((0, 0), window=inner_control_frame_cv, anchor="nw")

def on_cv_frame_configure(event):
    control_canvas_cv.configure(scrollregion=control_canvas_cv.bbox("all"))
inner_control_frame_cv.bind("<Configure>", on_cv_frame_configure)


# --- SLIDERS (Placed inside inner_control_frame_cv) ---

# Function to create a standardized slider control
def create_slider(parent, text, from_, to, initial, resolution=1):
    tk.Label(parent, text=text).pack(pady=2, fill='x')
    slider = tk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL, 
                      command=update_preview_grid, resolution=resolution)
    slider.set(initial)
    slider.pack(pady=5, fill='x')
    return slider

tk.Label(inner_control_frame_cv, text="--- Color & Light Filters ---", font=('Arial', 10, 'bold')).pack(pady=5)
slider_brightness = create_slider(inner_control_frame_cv, "Brightness (Beta: 0-200)", 0, 200, 100)
slider_contrast = create_slider(inner_control_frame_cv, "Contrast (Alpha: 0-200)", 0, 200, 100)
slider_gamma = create_slider(inner_control_frame_cv, "Gamma Correction (10-300)", 10, 300, 100, resolution=10)

slider_hue = create_slider(inner_control_frame_cv, "Hue Shift (-100 to +100)", 0, 200, 100)
slider_saturation = create_slider(inner_control_frame_cv, "Saturation Boost (0-200)", 0, 200, 100)
slider_value = create_slider(inner_control_frame_cv, "Value (Brightness in HSV)", 0, 200, 100)

# --- Image Geometry ---
tk.Label(inner_control_frame_cv, text="--- Image Geometry ---", font=('Arial', 10, 'bold')).pack(pady=5)

# 1. Pixel Dimensions Label
lbl_geometry_text = tk.Label(inner_control_frame_cv, text="Size: N/A", fg="blue", font=('Arial', 9))
lbl_geometry_text.pack(pady=0)

# 2. Geometry State Variables
var_square_padding = tk.BooleanVar(value=False)
var_high_quality_downscale = tk.BooleanVar(value=True) # Default to True (it's usually better)

# 3. Unified Update Logic
def on_geometry_change(*args):
    val = slider_scale_factor.get()
    
    if loaded_files_list:
        h, w = loaded_files_list[0]['bgr'].shape[:2]
        scale = float(val) / 100.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if var_square_padding.get():
            max_dim = max(new_w, new_h)
            lbl_geometry_text.config(text=f"{max_dim} x {max_dim} px (Squared)")
        else:
            lbl_geometry_text.config(text=f"{new_w} x {new_h} px")
    else:
        lbl_geometry_text.config(text="Load an Image")

    update_preview_grid()

# 4. Slider
tk.Label(inner_control_frame_cv, text="Scale Factor (%)").pack(pady=2, fill='x')
slider_scale_factor = tk.Scale(inner_control_frame_cv, from_=1, to=200, orient=tk.HORIZONTAL, 
                               command=on_geometry_change, resolution=1)
slider_scale_factor.set(100)
slider_scale_factor.pack(pady=5, fill='x')

# 5. Toggles
tk.Checkbutton(inner_control_frame_cv, text="Square Image (Padding)", variable=var_square_padding,
               command=on_geometry_change, anchor='w').pack(pady=2, fill='x')

tk.Checkbutton(inner_control_frame_cv, text="High-Quality Downscale (Area)", variable=var_high_quality_downscale,
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

# Flipping
var_flip_h = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Flip Horizontal", variable=var_flip_h, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_flip_v = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Flip Vertical", variable=var_flip_v, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')


tk.Label(inner_control_frame_cv, text="--- Blur Filters ---", font=('Arial', 10, 'bold')).pack(pady=5)
slider_gaussian_blur = create_slider(inner_control_frame_cv, "Gaussian Blur (Kernel Size)", 1, 25, 1, resolution=2)
slider_pixelation_block_size = create_slider(inner_control_frame_cv, "Pixelation Block Size (1=Off)", 1, 25, 1)


# --- BLUR & GRAYSCALE TOGGLES ---
tk.Label(inner_control_frame_cv, text="--- Image Toggles ---", font=('Arial', 10, 'bold')).pack(pady=5)

var_grayscale = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Grayscale Mode", variable=var_grayscale, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_median_blur = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Median Blur (5x5 Denoise)", variable=var_median_blur, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_bilateral = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Bilateral Filter (Edge-Preserving Denoise)", variable=var_bilateral, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_box_blur = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Box Blur (Simple Average)", variable=var_box_blur, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_sharpen = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Sharpening Filter", variable=var_sharpen, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_equalize_hist = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Auto Contrast (Equalize Hist)", variable=var_equalize_hist, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_invert = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Invert Colors (Negative)", variable=var_invert, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_threshold = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Binary Threshold (B&W)", variable=var_threshold, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_sepia = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Sepia Tone Filter", variable=var_sepia, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_channel_swap = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Color Channel Swap (R-B)", variable=var_channel_swap, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')


# --- Aesthetic Filters ---
tk.Label(inner_control_frame_cv, text="--- Aesthetic Filters ---", font=('Arial', 10, 'bold')).pack(pady=5)

var_cartoon_filter = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Cartoon (Stylization) Filter", variable=var_cartoon_filter, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_posterization_filter = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Posterization (Color Quantization)", variable=var_posterization_filter, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

slider_poster_colors = create_slider(inner_control_frame_cv, "Poster Colors (K-Means)", 2, 16, 8, resolution=1)


# --- MORPHOLOGICAL FILTERS ---
tk.Label(inner_control_frame_cv, text="--- Morphological Filters (Struct.) ---", font=('Arial', 10, 'bold')).pack(pady=5)

var_erode = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Erosion (Shrink Features)", variable=var_erode, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')

var_dilate = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Dilation (Expand Features)", variable=var_dilate, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')


# --- EDGE DETECTION RADIO BUTTONS ---
tk.Label(inner_control_frame_cv, text="--- Edge Detection (Select One) ---", font=('Arial', 10, 'bold')).pack(pady=5)

var_edge_mode = tk.StringVar(value="None")

edge_modes = {
    "None": "Off",
    "Canny": "Canny (Thresholded)",
    "Laplacian": "Laplacian",
    "SobelComplete": "Sobel X+Y (Combined)",
    "SobelX": "Sobel X (Vertical Edges)",
    "SobelY": "Sobel Y (Horizontal Edges)",
}

for mode, text in edge_modes.items():
    tk.Radiobutton(inner_control_frame_cv, text=text, variable=var_edge_mode, value=mode, 
                   command=update_preview_grid, anchor='w').pack(pady=1, fill='x')
    
# --- CUSTOM COLOR TINT ---
tk.Label(inner_control_frame_cv, text="--- Custom Hue Tint ---", font=('Arial', 10, 'bold')).pack(pady=5)

tk.Button(inner_control_frame_cv, text="Select Tint Color", command=select_tint_color, 
          bg='#F0F0F0').pack(pady=2, fill='x')

var_custom_tint_toggle = tk.BooleanVar(value=False)
tk.Checkbutton(inner_control_frame_cv, text="Apply Custom Tint", variable=var_custom_tint_toggle, 
               command=update_preview_grid, anchor='w').pack(pady=2, fill='x')


# --- IMAGE COMPARISON ---
tk.Label(inner_control_frame_cv, text="--- Image Comparison (MSE) ---", font=('Arial', 10, 'bold')).pack(pady=5)
tk.Button(inner_control_frame_cv, text="Load Comparison Image", command=load_compare_image, 
          bg='#F0F0F0').pack(pady=2, fill='x')
tk.Button(inner_control_frame_cv, text="Calculate MSE", command=compare_action, 
          bg='#D6EAF8').pack(pady=2, fill='x')


# --- Tab 2: AI Image-to-Image ---
tab_ai_reimagine = tk.Frame(notebook)
notebook.add(tab_ai_reimagine, text='AI Image-to-Image')

# AI Reimagine Control Setup
tk.Label(tab_ai_reimagine, text="Gemini Image-to-Image Reimagining", font=('Arial', 12, 'bold')).pack(pady=10)

# --- NEW: Style Preset Selection ---
style_container = tk.LabelFrame(tab_ai_reimagine, text="Style Presets", padx=10, pady=10)
style_container.pack(pady=10, padx=10, fill='x')

tk.Label(style_container, text="Select vibe:").pack(side=tk.LEFT)
var_style_selection = tk.StringVar(value="None")
style_dropdown = ttk.Combobox(style_container, textvariable=var_style_selection, 
                              values=list(STYLE_PRESETS.keys()), state="readonly")
style_dropdown.pack(side=tk.LEFT, padx=5)

# Optional: Add the randomize button right next to the dropdown
#tk.Button(style_container, text="🎲", command=randomize_style).pack(side=tk.LEFT)
# ----------------------------------

tk.Label(tab_ai_reimagine, text="1. Source must be the first loaded image.").pack(pady=5)
tk.Label(tab_ai_reimagine, text="2. Enter Prompt:", anchor='w').pack(pady=5, fill='x')

ai_prompt_entry_reimagine = tk.Text(tab_ai_reimagine, height=6, width=40)
ai_prompt_entry_reimagine.pack(padx=10, pady=5, fill='x', expand=False)
ai_prompt_entry_reimagine.insert(tk.END, "A high-quality oil painting of this image.")

tk.Label(tab_ai_reimagine, text="3. Generate:", anchor='w').pack(pady=5, fill='x')

ai_loading_label = tk.Label(main_control_frame, text="Ready.", fg='black') 
ai_loading_label.pack(in_=tab_ai_reimagine, pady=5)

ai_reimagine_button = tk.Button(tab_ai_reimagine, text="Reimagine Image (Gemini 2.5 Flash)", 
                                command=reimagine_image_action, bg='#F7DC6F')
ai_reimagine_button.pack(pady=10, padx=10, fill='x')


# --- Tab 3: AI Image Generation (Text-to-Image) ---
tab_ai_generate = tk.Frame(notebook)
notebook.add(tab_ai_generate, text='AI Image Generation')

# AI Generate Control Setup
tk.Label(tab_ai_generate, text="Imagen Text-to-Image Generation", font=('Arial', 12, 'bold')).pack(pady=10)
tk.Label(tab_ai_generate, text="1. Enter Prompt for new image generation:").pack(pady=5)

ai_prompt_entry_generate = tk.Text(tab_ai_generate, height=6, width=40)
ai_prompt_entry_generate.pack(padx=10, pady=5, fill='x', expand=False)
ai_prompt_entry_generate.insert(tk.END, "A majestic golden retriever wearing a tiny crown, digital art.")

tk.Label(tab_ai_generate, text="2. Number of Regenerations (1-10):", anchor='w').pack(pady=5, fill='x')
ai_num_generations = tk.Entry(tab_ai_generate)
ai_num_generations.insert(0, "1")
ai_num_generations.pack(padx=10, pady=5, fill='x')

tk.Label(tab_ai_generate, text="3. Generate:", anchor='w').pack(pady=5, fill='x')

# Repack the status label for Tab 3 
# We need to explicitly unpack the label before moving it to the new parent's pack
# This is safe because it uses pack_forget() which handles the geometry manager state.
ai_loading_label.pack_forget() 
ai_loading_label.pack(in_=tab_ai_generate, pady=5) # Pack into Tab 3

ai_generate_button = tk.Button(tab_ai_generate, text="Generate New Image (Imagen 4.0)", command=generate_new_image_action, 
          bg='#D9B2EE')
ai_generate_button.pack(pady=10, padx=10, fill='x')


# --- ACTIONS FRAME (Outside the Notebook for persistence) ---
actions_frame = tk.Frame(main_control_frame)
actions_frame.pack(side=tk.BOTTOM, fill='x', padx=5, pady=10)

# Reset Button
tk.Button(actions_frame, text="Reset Filters", command=lambda: reset_sliders(True), 
          bg='lightcoral').pack(pady=5, fill='x')

# File Management Buttons
tk.Button(actions_frame, text="Load New Image(s)", command=load_image, 
          bg='lightgreen').pack(pady=5, fill='x')
tk.Button(actions_frame, text="Save Processed Image", command=save_image, 
          bg='orange').pack(pady=5, fill='x')


# --- Image Display Area (Right Side, now a dynamic grid) ---
display_frame = tk.Frame(root)
display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Main display canvas (used for scrolling the image grid)
control_canvas_display = tk.Canvas(display_frame, bg='gray')
control_canvas_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Scrollbar for the image grid
scroll_v_display = tk.Scrollbar(display_frame, orient=tk.VERTICAL, command=control_canvas_display.yview)
scroll_v_display.pack(side=tk.RIGHT, fill=tk.Y)
control_canvas_display.config(yscrollcommand=scroll_v_display.set)

# Frame for the grid itself, placed inside the display canvas
image_grid_frame = tk.Frame(control_canvas_display, padx=10, pady=10)
control_canvas_display.create_window((0, 0), window=image_grid_frame, anchor="nw")

# Update canvas scroll region when the inner frame changes size
def on_grid_frame_configure(event):
    control_canvas_display.config(scrollregion=control_canvas_display.bbox("all"))

image_grid_frame.bind("<Configure>", on_grid_frame_configure)

# --- Initial Load and Run ---
root.mainloop()