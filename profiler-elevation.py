import requests
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

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

def detect_marine_layer_depth(rgb_columns, base_alt_ft=0, top_alt_ft=5000, detection_max_ft=3500, show=False):
    height = rgb_columns.shape[0]
    altitudes = np.linspace(top_alt_ft, base_alt_ft, height)
    avg_column = rgb_columns.mean(axis=1).astype(int)

    if show:
        fig, ax = plt.subplots(figsize=(2, 6))
        ax.imshow(avg_column.reshape(-1, 1, 3), aspect='auto', extent=[0, 1, base_alt_ft, top_alt_ft])
        ax.set_ylabel("Altitude (ft)")
        ax.set_xticks([])
        ax.set_yticks(np.arange(base_alt_ft, top_alt_ft + 1, 500))
        ax.invert_yaxis()
        ax.set_title("Most Recent Profiler Column")
        plt.tight_layout()
        plt.show()

    print("\n--- RGB Classification (0–3500 ft) ---")
    in_marine = False
    marine_start_index = None

    for i in range(height - 1, -1, -1):  # bottom (surface) to top
        alt = int(altitudes[i])
        if alt > detection_max_ft:
            continue

        rgb = avg_column[i]
        label = classify_color(rgb)
        print(f"{alt:5d} ft: {tuple(rgb)} => {label}")

        if label == "marine":
            if not in_marine:
                in_marine = True
                marine_start_index = i
        elif in_marine:
            # First non-marine pixel after marine zone
            transition_alt = int(altitudes[i + 1])
            print(f"\n✅ Detected marine layer top at {transition_alt} ft")
            return transition_alt

    if in_marine and marine_start_index is not None:
        fallback_alt = int(altitudes[marine_start_index])
        print(f"\n✅ Fallback: marine layer top at {fallback_alt} ft (no transition found)")
        return fallback_alt

    print(f"\n⚠️ No marine layer detected below {detection_max_ft} ft.")
    return None

def main():
    url = "https://met.nps.edu/~lind/profiler/ord_mix.gif"
    image = download_profiler_image(url)
    latest_columns = extract_latest_column(image)
    marine_layer_depth = detect_marine_layer_depth(latest_columns, detection_max_ft=3500)

#    Old output
#    if marine_layer_depth is not None:
#        print(f"Estimated marine layer depth: {marine_layer_depth} feet")
#    else:
#        print("Unable to detect marine layer depth.")

if __name__ == "__main__":
    main()
