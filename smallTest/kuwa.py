import cv2
import numpy as np

def sobel_gradients(img):
    # compute sobel gradients in x and y directions
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def compute_dog(img):
    # compute difference of gaussians (dog) for edge enhancement
    gaussian_blur1 = cv2.GaussianBlur(img, (5, 5), 0.5)
    gaussian_blur2 = cv2.GaussianBlur(img, (5, 5), 1)
    return gaussian_blur1 - gaussian_blur2

def kuwahara_local_filter(img, x, y, radius):
    half_radius = radius // 2
    h, w = img.shape[:2]

    # define regions for kuwahara filter
    def safe_slice(x_start, x_end, y_start, y_end):
        # ensure indices are within bounds
        x_start = max(0, x_start)
        x_end = min(h, x_end)
        y_start = max(0, y_start)
        y_end = min(w, y_end)
        return img[x_start:x_end, y_start:y_end]

    regions = {
        '1': safe_slice(x - half_radius, x + 1, y - half_radius, y + 1),
        '2': safe_slice(x - half_radius, x + 1, y, y + half_radius + 1),
        '3': safe_slice(x, x + half_radius + 1, y - half_radius, y + 1),
        '4': safe_slice(x, x + half_radius + 1, y, y + half_radius + 1)
    }

    def compute_variance(region):
        # ensure the region has more than one pixel
        if region.size > 1:
            return np.var(region, ddof=1)
        else:
            return 0

    # compute variances for each region
    variances = {key: compute_variance(region) for key, region in regions.items()}

    # find the region with the minimum variance
    min_variance_key = min(variances, key=variances.get)
    return np.mean(regions[min_variance_key], axis=(0, 1))

def anisotropic_kuwahara(img, window_size=7):
    # split the image into its color channels
    channels = cv2.split(img)
    filtered_channels = []

    for ch in channels:
        filtered_ch = np.zeros_like(ch, dtype=np.float32)
        radius = window_size // 2

        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                filtered_ch[i, j] = kuwahara_local_filter(ch, i, j, window_size)

                # print progress every 1000 pixels to check for infinite loops
                if i % 100 == 0 and j % 100 == 0:
                    print(f"processing pixel ({i}, {j})")

        filtered_channels.append(np.clip(filtered_ch, 0, 255).astype(np.uint8))

    # merge the filtered color channels back into a single image
    filtered_img = cv2.merge(filtered_channels)
    return filtered_img

def main():
    # load an image from file
    img = cv2.imread('flight.jpg')  # load color image
    if img is None:
        print("error loading image")
        return

    # resize the image for faster processing
    scale_factor = 0.8  # adjust the scale factor as needed
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # convert to float32 for processing
    img = img.astype(np.float32)

    # apply anisotropic kuwahara filter
    window_size = 7  # you can adjust the window size here
    print("starting filter application...")
    filtered_img = anisotropic_kuwahara(img, window_size)
    print("filter application complete.")

    # save the result
    cv2.imwrite('output_image.jpg', filtered_img)

    # display the results
    cv2.imshow('original image', img.astype(np.uint8))
    cv2.imshow('filtered image', filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
