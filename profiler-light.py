import requests
from PIL import Image
import numpy as np
import io

def download_profiler_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

def extract_latest_column(image, num_cols=5, data_start_x=135):
    image_rgb = image.convert("RGB")
    width, height = image.size
    left = data_start_x
    right = data_start_x + num_cols
    data_column = image_rgb.crop((left, 0, right, height))
    return np.array(data_column).reshape((height, num_cols, 3))

def classify_color(rgb):
    r, g, b = rgb
    total = r + g + b + 1e-5
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total

    if b_norm > 0.45 and b > 50:
        return "marine"
    elif r_norm > 0.35 and g_norm > 0.35 and r > 90 and g > 90:
        return "inversion"
    else:
        return "other"

def detect_marine_layer_depth(
    rgb_columns, base_alt_ft=0, top_alt_ft=5000, detection_max_ft=3500,
    required_nonmarine=3
):
    height = rgb_columns.shape[0]
    altitudes = np.linspace(top_alt_ft, base_alt_ft, height)
    avg_column = rgb_columns.mean(axis=1).astype(int)

    in_marine = False
    nonmarine_streak = 0

    for i in range(height - 1, -1, -1):  # bottom (surface) to top
        alt = int(altitudes[i])
        if alt > detection_max_ft:
            continue

        rgb = avg_column[i]
        label = classify_color(rgb)

        if label == "marine":
            in_marine = True
            nonmarine_streak = 0
        elif in_marine:
            nonmarine_streak += 1
            if nonmarine_streak >= required_nonmarine:
                return int(altitudes[i + required_nonmarine])

    if in_marine:
        return int(altitudes[i + 1]) if i + 1 < height else int(altitudes[-1])

    return None

def main():
    url = "https://met.nps.edu/~lind/profiler/ord_mix.gif"
    image = download_profiler_image(url)
    latest_columns = extract_latest_column(image)
    marine_layer_depth = detect_marine_layer_depth(latest_columns, detection_max_ft=3500)

    if marine_layer_depth is not None:
        print(marine_layer_depth)

if __name__ == "__main__":
    main()
