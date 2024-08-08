from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

# Define the grayscale gradient (from least dense to most dense)
gscale = np.asarray(list("c.oar=+*%@#%&8@#"))

# Edge characters
edge_chars = {
    'h': '-',
    'v': '|',
    'r': '/',
    'l': '\\'
}

def gaussian_filter(image_array, sigma=1):
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size)
    )
    kernel /= np.sum(kernel)

    pad_size = size // 2
    padded_image = np.pad(image_array, pad_size, mode='constant', constant_values=0)
    smooth_img = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            smooth_img[i, j] = np.sum(kernel * padded_image[i:i + size, j:j + size])
    return smooth_img

def difference_of_gaussians(image_array, sigma1=1, sigma2=2):
    dog1 = gaussian_filter(image_array, sigma1)
    dog2 = gaussian_filter(image_array, sigma2)
    return dog1 - dog2

def sobel_filter(image_array):
    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    pad_size = 1
    padded_image = np.pad(image_array, pad_size, mode='constant', constant_values=0)
    edges = np.zeros_like(image_array, dtype=float)
    gx = np.zeros_like(image_array, dtype=float)
    gy = np.zeros_like(image_array, dtype=float)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            region = padded_image[i:i + 3, j:j + 3]
            gx[i, j] = np.sum(sobel_x * region)
            gy[i, j] = np.sum(sobel_y * region)
            edges[i, j] = np.hypot(gx[i, j], gy[i, j])

    return edges, gx, gy

def non_maximum_suppression(edges, gx, gy):
    directions = np.arctan2(gy, gx) * (180 / np.pi)
    directions[directions < 0] += 180
    nms = np.zeros_like(edges, dtype=np.uint8)

    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            angle = directions[i, j]
            q = 255
            r = 255

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = edges[i, j + 1]
                r = edges[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = edges[i + 1, j - 1]
                r = edges[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = edges[i + 1, j]
                r = edges[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = edges[i - 1, j - 1]
                r = edges[i + 1, j + 1]

            if (edges[i, j] >= q) and (edges[i, j] >= r):
                nms[i, j] = edges[i, j]
            else:
                nms[i, j] = 0

    return nms

def map_to_ascii(image_array, gscale):
    ascii_array = (image_array / 255 * (len(gscale) - 1)).astype(int)
    return gscale[ascii_array]

def apply_edges_to_ascii(ascii_art, edges, directions, edge_chars):
    for y in range(ascii_art.shape[0]):
        for x in range(ascii_art.shape[1]):
            if edges[y, x] > 100:
                angle = directions[y, x]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    ascii_art[y, x] = edge_chars['h']
                elif 67.5 <= angle < 112.5:
                    ascii_art[y, x] = edge_chars['v']
                elif 22.5 <= angle < 67.5:
                    ascii_art[y, x] = edge_chars['r']
                elif 112.5 <= angle < 157.5:
                    ascii_art[y, x] = edge_chars['l']
    return ascii_art

def adjust_brightness(color, factor):
    return tuple(min(int(c * factor), 255) for c in color)

def get_grayscale_color(char, gscale):
    if char in gscale:
        index = np.where(gscale == char)[0][0]
        grayscale_value = index / (len(gscale) - 1) * 255
        return (int(grayscale_value), int(grayscale_value), int(grayscale_value))
    return (0, 0, 0)  # Default to black if character not in gscale

def apply_depth_map(ascii_art, depth_map, gscale, edge_chars):
    depth_normalized = depth_map / 255  # Normalize depth map to [0, 1]
    new_ascii_art = np.copy(ascii_art)

    for y in range(ascii_art.shape[0]):
        for x in range(ascii_art.shape[1]):
            depth_value = depth_normalized[y, x]
            char = ascii_art[y, x]

            if char in edge_chars.values():
                continue

            # Retrieve the original color of the ASCII character
            original_color = get_grayscale_color(char, gscale)
            # Calculate the brightness adjustment factor based on depth value
            brightness_factor = 1 + (0.5 * depth_value)  # Scale factor to avoid blowing out

            # Adjust the brightness of the color
            new_color = adjust_brightness(original_color, brightness_factor)
            # Find the closest ASCII character after brightness adjustment
            new_char = min(gscale, key=lambda ch: np.abs(int(np.mean(get_grayscale_color(ch, gscale))) - np.mean(new_color)))
            new_ascii_art[y, x] = new_char

    return new_ascii_art

def draw_ascii_image(ascii_art, img_rgb_array, char_size):
    ascii_img_width = char_size * ascii_art.shape[1]
    ascii_img_height = char_size * ascii_art.shape[0]

    ascii_img = Image.new('RGB', (ascii_img_width, ascii_img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(ascii_img)

    font = ImageFont.load_default()

    for y in range(ascii_art.shape[0]):
        for x in range(ascii_art.shape[1]):
            char = ascii_art[y, x]
            color = tuple(img_rgb_array[y, x])
            draw.text((x * char_size, y * char_size), char, font=font, fill=color)

    return ascii_img

def save_image_with_title(img, title, filename):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Load and preprocess the image
img_title = 'out-2.png'
img = Image.open(img_title)
img_rgb = img
img_gray = img.convert('L')
width, height = img.size
scale = 4
nscale = (width // scale, height // scale)
img_gray = img_gray.resize(nscale)
img_rgb = img_rgb.resize(nscale)

# Load and preprocess the depth map
depth_map = Image.open('out-2depth.png').convert('L')
depth_map = depth_map.resize(nscale)
depth_map_array = np.asarray(depth_map)

# Convert the grayscale image to a numpy array
img_array = np.asarray(img_gray)

# Display the pop-up window for adjusting sigma values
sigma1_default = 0.5
sigma2_default = 1

def update_dog(val):
    sigma1 = float(text_sigma1.text)
    sigma2 = float(text_sigma2.text)
    dog_img = difference_of_gaussians(img_array, sigma1, sigma2)
    ax_dog.imshow(dog_img, cmap='gray')
    ax_dog.set_title(f'Difference of Gaussians (sigma1={sigma1}, sigma2={sigma2})')
    plt.draw()

def apply_and_close(event):
    sigma1 = float(text_sigma1.text)
    sigma2 = float(text_sigma2.text)
    global final_sigma1, final_sigma2
    final_sigma1, final_sigma2 = sigma1, sigma2
    plt.close()

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.4)

ax_dog = plt.axes([0.1, 0.5, 0.8, 0.4], aspect='equal')
dog_img = difference_of_gaussians(img_array, sigma1_default, sigma2_default)
ax_dog.imshow(dog_img, cmap='gray')
ax_dog.set_title(f'Difference of Gaussians (sigma1={sigma1_default}, sigma2={sigma2_default})')
ax_dog.axis('off')

text_sigma1 = TextBox(plt.axes([0.1, 0.35, 0.3, 0.05]), 'Sigma1', initial=str(sigma1_default))
text_sigma2 = TextBox(plt.axes([0.1, 0.3, 0.3, 0.05]), 'Sigma2', initial=str(sigma2_default))

update_button = Button(plt.axes([0.6, 0.35, 0.3, 0.05]), 'Update')
update_button.on_clicked(update_dog)

apply_button = Button(plt.axes([0.6, 0.3, 0.3, 0.05]), 'Apply and Close')
apply_button.on_clicked(apply_and_close)

plt.show()

# Apply final DoG with adjusted sigma values
dog_img = difference_of_gaussians(img_array, final_sigma1, final_sigma2)
edges, gx, gy = sobel_filter(dog_img)
nms_edges = non_maximum_suppression(edges, gx, gy)

# Convert the grayscale image to ASCII art
ascii_art = map_to_ascii(img_array, gscale)

# Apply edges to ASCII art
ascii_art_with_edges = apply_edges_to_ascii(ascii_art, nms_edges, np.arctan2(gy, gx) * (180 / np.pi), edge_chars)

# Apply depth map adjustment
ascii_art_adjusted = apply_depth_map(ascii_art_with_edges, depth_map_array, gscale, edge_chars)

# Convert the original image to RGB and resize for drawing
img_rgb_array = np.asarray(img_rgb)

# Define character block size
char_size = 12

# Draw ASCII art images
ascii_img_before_depth_map = draw_ascii_image(ascii_art, img_rgb_array, char_size)
ascii_img_after_depth_map = draw_ascii_image(ascii_art_adjusted, img_rgb_array, char_size)

# Save ASCII art images
ascii_img_before_depth_map.save(img_title + ' ascii_art_before_depth_map.png')
ascii_img_after_depth_map.save(img_title + ' ascii_art_after_depth_map.png')

# Display images in the specified order
def display_image(image, title):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display images one at a time
display_image(img_rgb, 'Original Image')
display_image(dog_img, 'Difference of Gaussians (DoG)')
display_image(edges, 'Sobel Edges')
display_image(ascii_img_before_depth_map, 'ASCII Art Before Depth Map')
display_image(ascii_img_after_depth_map, 'ASCII Art with Depth Map')
