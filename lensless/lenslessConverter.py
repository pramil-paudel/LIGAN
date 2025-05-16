import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.signal import max_len_seq

numScenePix = 64
numMaskPix = 63
unitMask = 2
numSensorPix = 128
unitSensor = 15
d = 40000  # sensor to mask distance
midScenePix = int(np.floor(numScenePix / 2))
midSensorPix = int(np.floor(numSensorPix / 2)) - 1
# helper functions
z2a = lambda z: (1 - d / z)
a2z = lambda a: d / (1 - a)
crop_sensor = lambda x: x[:, midScenePix: midScenePix + numSensorPix, midScenePix:midScenePix + numSensorPix]
crop_img = lambda x: x[:, midSensorPix: midSensorPix + numScenePix, midSensorPix: midSensorPix + numScenePix]
# create a list of predefined depth planes
depthList = z2a(np.array([50e4]))  # 50cm away   (100000:1km, 50000:500m, 100:1m)
# define functions for interpolation
locSensor = np.linspace(-numSensorPix * unitSensor / 2 + unitSensor / 2, numSensorPix * unitSensor / 2 - unitSensor / 2,
                        numSensorPix)
locMask = np.linspace(-numMaskPix * unitMask / 2 + unitMask / 2, numMaskPix * unitMask / 2 - unitMask / 2, numMaskPix)
std = 0.02


def create_mask_for_lensless():
    def generateInterpMatrix(depthList, locSensor, locMask):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])
        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)
            interpMatrix[np.arange(len(lo)), lo, di] = 1 - (locInterp - locMask[lo]) / unitMask
            interpMatrix[np.arange(len(hi)), hi, di] = 1 - (locMask[hi] - locInterp) / unitMask
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0
        return interpMatrix

    # ------------------------------------------------------------------------------
    # create mask pattern
    maskVec = max_len_seq(int(np.log2(numMaskPix + 1)))[0].reshape(numMaskPix, 1)
    maskPattern = maskVec @ maskVec.T
    # print(maskPattern)
    # generate interpolation matrices for multiple depths
    interpMatrix = generateInterpMatrix(depthList, locSensor, locMask)[:, :, 0]
    psf = interpMatrix @ maskPattern @ interpMatrix.T
    return crop_img, crop_sensor, psf


def create_non_binary_mask_for_lensless():
    # linear interpolation matrix
    def generateInterpMatrix(depthList, locSensor, locMask):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])
        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)
            # Here we modify the weights to be non-binary (smooth interpolation)
            interpMatrix[np.arange(len(lo)), lo, di] = 1 - (locInterp - locMask[lo]) / unitMask
            interpMatrix[np.arange(len(hi)), hi, di] = 1 - (locMask[hi] - locInterp) / unitMask

            # Smoothen the edges for the interpolation
            interpMatrix[np.arange(len(lo)), lo, di] = np.clip(interpMatrix[np.arange(len(lo)), lo, di], 0, 1)
            interpMatrix[np.arange(len(hi)), hi, di] = np.clip(interpMatrix[np.arange(len(hi)), hi, di], 0, 1)
            # Handling values outside the mask range
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0
        return interpMatrix

    # ------------------------------------------------------------------------------
    # Create mask pattern
    maskVec = max_len_seq(int(np.log2(numMaskPix + 1)))[0].reshape(numMaskPix, 1)
    maskPattern = maskVec @ maskVec.T

    # Generate interpolation matrices for multiple depths
    interpMatrix = generateInterpMatrix(depthList, locSensor, locMask)[:, :, 0]

    # Now the PSF will be non-binary
    psf = interpMatrix @ maskPattern @ interpMatrix.T

    return crop_img, crop_sensor, psf


def create_gaussian_weighted_psf():
    numSensorPix = 128
    numMaskPix = 128

    # Gaussian-weighted interpolation matrix
    def generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix, sigma=0.5):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])

        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)

            # Ensure indices stay within bounds
            lo = np.clip(lo, 0, len(locMask) - 1)
            hi = np.clip(hi, 0, len(locMask) - 1)

            valid_indices = np.arange(min(len(lo), numSensorPix))

            interpMatrix[valid_indices, lo[valid_indices], di] = 1 - (
                    locInterp[valid_indices] - locMask[lo[valid_indices]]) / unitMask
            interpMatrix[valid_indices, hi[valid_indices], di] = 1 - (
                    locMask[hi[valid_indices]] - locInterp[valid_indices]) / unitMask

            # Smoothen the edges for the interpolation
            interpMatrix[valid_indices, lo[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, lo[valid_indices], di], 0, 1)
            interpMatrix[valid_indices, hi[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, hi[valid_indices], di], 0, 1)

            # Handling values outside the mask range
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0

        return interpMatrix

    # Create a fixed but different mask pattern using a deterministic random seed
    np.random.seed(42)  # Ensures the pattern is different but fixed for each run
    maskPattern = np.random.randint(0, 2, (numMaskPix, numMaskPix))

    # Introduce variations in at least 900 locations
    random_indices = np.random.choice(numMaskPix * numMaskPix, 900, replace=False)
    maskPattern.flat[random_indices] = 1 - maskPattern.flat[random_indices]

    # Generate interpolation matrices for multiple depths
    interpMatrix = generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix)[:, :, 0]

    # Apply transformation to create the new PSF
    psf = interpMatrix @ maskPattern @ interpMatrix.T

    return crop_img, crop_sensor, psf


def create_mask_for_lensless_second_version():
    unitMask = 20

    # Gaussian-weighted interpolation matrix
    def generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix, sigma=0.5):
        numDepth = depthList.size
        interpMatrix = np.zeros([numSensorPix, numMaskPix, numDepth])

        for di in np.arange(numDepth):
            locInterp = depthList[di] * locSensor
            hi = np.minimum(len(locMask) - 1, np.searchsorted(locMask, locInterp, 'right'))
            lo = np.maximum(0, hi - 1)

            # Ensure indices stay within bounds
            lo = np.clip(lo, 0, len(locMask) - 1)
            hi = np.clip(hi, 0, len(locMask) - 1)

            valid_indices = np.arange(min(len(lo), numSensorPix))

            interpMatrix[valid_indices, lo[valid_indices], di] = 1 - (
                    locInterp[valid_indices] - locMask[lo[valid_indices]]) / unitMask
            interpMatrix[valid_indices, hi[valid_indices], di] = 1 - (
                    locMask[hi[valid_indices]] - locInterp[valid_indices]) / unitMask

            # Smoothen the edges for the interpolation
            interpMatrix[valid_indices, lo[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, lo[valid_indices], di], 0, 1)
            interpMatrix[valid_indices, hi[valid_indices], di] = np.clip(
                interpMatrix[valid_indices, hi[valid_indices], di], 0, 1)

            # Handling values outside the mask range
            rowsCast = np.where(np.logical_or(locInterp < locMask[0], locInterp > locMask[-1]))
            interpMatrix[rowsCast, :, di] = 0

        return interpMatrix

    # Create a fixed but different mask pattern using a deterministic random seed
    np.random.seed(42)  # Ensures the pattern is different but fixed for each run
    maskPattern = np.random.randint(0, 2, (numMaskPix, numMaskPix))

    # Introduce variations in at least 900 locations
    random_indices = np.random.choice(numMaskPix * numMaskPix, 900, replace=False)
    maskPattern.flat[random_indices] = 1 - maskPattern.flat[random_indices]

    # Generate interpolation matrices for multiple depths
    interpMatrix = generateGaussianInterpMatrix(depthList, locSensor, locMask, numSensorPix, numMaskPix)[:, :, 0]

    # Apply transformation to create the new PSF
    psf = interpMatrix @ maskPattern @ interpMatrix.T

    return crop_img, crop_sensor, psf


def convert_image_to_lensless(image):
    # This returns an image with noise and lensless transformation
    crop_img, crop_sensor, psf = create_mask_for_lensless()
    fullLength = numSensorPix + numScenePix - 1
    psf_fft = np.fft.fft2(psf, s=[fullLength, fullLength])
    image = image.resize((numScenePix, numScenePix)).convert("RGB")
    image = np.array(image)
    image = image.transpose(2, 0, 1).astype(np.double) / 255
    img_fft = np.fft.fft2(image, s=[fullLength, fullLength], axes=[1, 2])
    y_fft = psf_fft * img_fft
    y = crop_sensor(np.fft.ifft2(y_fft, axes=[1, 2]).real)
    y_im = y
    # Normalizing the image so that image disk writing makes little difference
    y_im -= np.amin(y_im, axis=(1, 2))[:, None, None]
    y_im /= np.amax(y_im, axis=(1, 2))[:, None, None]
    y_im = 255 * y_im
    y_im = y_im.transpose(1, 2, 0).astype(np.uint8)
    return y_im


def reconstruct_an_image_different_binary_pattern(image):
    image = np.array(image)
    image = image.transpose(2, 0, 1).astype(np.uint8)
    crop_img, crop_sensor, psf = create_mask_for_lensless_second_version()
    fullLength = numSensorPix + numScenePix - 1
    psf_fft = np.fft.fft2(psf, s=[fullLength, fullLength])
    lambda_snr = np.sqrt(std)
    xhat_fft = (np.conjugate(psf_fft) * np.fft.fft2(image, s=[fullLength, fullLength], axes=[1, 2])) / (
            np.abs(psf_fft) ** 2 + lambda_snr)
    xhat = crop_img(np.fft.fftshift(np.fft.ifft2(xhat_fft, axes=[1, 2]).real, axes=[1, 2]))
    all_zero = not np.any(xhat)
    if not all_zero:
        xhat -= np.amin(xhat, axis=(1, 2))[:, None, None]
        xhat /= np.amax(xhat, axis=(1, 2))[:, None, None]
    xhat = xhat * 255
    xhat = xhat.transpose(1, 2, 0)
    return xhat


import numpy as np


def pad_psf_to_match_image(psf, image_size):
    """
    Pads the PSF to match the image size.

    :param psf: Input PSF (Point Spread Function) as NumPy array
    :param image_size: Tuple (H, W) of the target image
    :return: Padded PSF
    """
    h, w = image_size  # Target size
    ph, pw = psf.shape  # PSF size

    # Create zero-padded PSF with same center alignment
    pad_h = (h - ph) // 2
    pad_w = (w - pw) // 2

    psf_padded = np.pad(psf, ((pad_h, h - ph - pad_h), (pad_w, w - pw - pad_w)), mode='constant', constant_values=0)
    return psf_padded


def apply_low_pass_filter(psf_fft, image_size):
    """
    Apply an adaptive Gaussian low-pass filter in the frequency domain.

    :param psf_fft: FFT of the PSF
    :param image_size: Tuple (H, W) of the image
    :return: Filtered PSF in frequency domain
    """
    h, w = image_size
    center = (h // 2, w // 2)

    # Dynamically set sigma based on image size (prevents over-smoothing)
    sigma = max(h, w) // 50  # Adapt to image resolution

    # Create Gaussian filter in the frequency domain
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    gaussian_filter = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma ** 2))

    return psf_fft * gaussian_filter


def reconstruct_an_image(image):
    image = np.array(image)
    image = image.transpose(2, 0, 1).astype(np.uint8)
    crop_img, crop_sensor, psf = create_mask_for_lensless()
    fullLength = numSensorPix + numScenePix - 1
    psf_fft = np.fft.fft2(psf, s=[fullLength, fullLength])
    lambda_snr = np.sqrt(std)
    xhat_fft = (np.conjugate(psf_fft) * np.fft.fft2(image, s=[fullLength, fullLength], axes=[1, 2])) / (
            np.abs(psf_fft) ** 2 + lambda_snr)
    xhat = crop_img(np.fft.fftshift(np.fft.ifft2(xhat_fft, axes=[1, 2]).real, axes=[1, 2]))
    all_zero = not np.any(xhat)
    if not all_zero:
        xhat -= np.amin(xhat, axis=(1, 2))[:, None, None]
        xhat /= np.amax(xhat, axis=(1, 2))[:, None, None]
    xhat = xhat * 255
    xhat = xhat.transpose(1, 2, 0)
    return xhat


def estimate_noise(image):
    """
    Estimate noise level using Laplacian variance.
    """
    image = np.array(image)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return np.var(cv2.Laplacian(image, cv2.CV_64F))


def pad_psf(psf, target_size):
    """
    Pads the PSF to match the image size before performing FFT operations.
    """
    h, w = target_size
    ph, pw = psf.shape
    pad_h = (h - ph) // 2
    pad_w = (w - pw) // 2
    return np.pad(psf, ((pad_h, h - ph - pad_h), (pad_w, w - pw - pad_w)), mode='constant')


def safe_normalize(xhat):
    """
    Normalizes image safely by avoiding division by zero.
    """
    min_val = np.amin(xhat, axis=(1, 2), keepdims=True)
    max_val = np.amax(xhat, axis=(1, 2), keepdims=True)
    max_val[max_val == 0] = 1
    return (xhat - min_val) / max_val


def wiener_filter(psf_fft, image_fft, reg_factor=0.01):
    """
    Applies Wiener filtering in the frequency domain.
    """
    psf_fft_conj = np.conjugate(psf_fft)
    power_spectrum = np.abs(psf_fft) ** 2

    return (psf_fft_conj * image_fft) / (power_spectrum + reg_factor + 1e-8)  # 🔥 Add small epsilon


def reconstruct_an_image_fixed(image, sigma=100, lambda_scale=1):
    image = np.array(image).transpose(2, 0, 1).astype(np.float32)

    crop_img, crop_sensor, psf = create_mask_for_lensless()
    fullLength = numSensorPix + numScenePix - 1

    psf_fft = np.fft.fft2(psf, s=[fullLength, fullLength])

    h, w = psf_fft.shape
    center_x, center_y = h // 2, w // 2
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    gaussian_filter = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    psf_fft_filtered = psf_fft * gaussian_filter

    lambda_snr = np.sqrt(std) * lambda_scale

    image_fft = np.fft.fft2(image, s=[fullLength, fullLength], axes=[1, 2])
    xhat_fft = (np.conjugate(psf_fft_filtered) * image_fft) / (np.abs(psf_fft_filtered) ** 2 + lambda_snr)

    xhat_ifft = np.fft.ifft2(xhat_fft, axes=[1, 2]).real
    xhat_shifted = np.fft.fftshift(xhat_ifft, axes=[1, 2])
    xhat = crop_img(xhat_shifted)

    if np.any(xhat):
        xhat -= np.min(xhat, axis=(1, 2), keepdims=True)
        xhat /= (np.max(xhat, axis=(1, 2), keepdims=True) + 1e-6)

    xhat = (xhat * 255).astype(np.uint8).transpose(1, 2, 0)
    return xhat


def reconstruct_individual_image_tensor(image):
    with torch.no_grad():
        data = image.numpy()
        image_c = np.array(data)
        print(image_c.shape)
        image_c = image_c.transpose(1, 2, 0)
        image_c = reconstruct_an_image(image_c)
    return image_c


def reconstruct_individual_image(image):
    with torch.no_grad():
        image_c = np.array(image)
        image_c = Image.fromarray(image_c, "RGB")
        image_c.save("this_is_input.png")
        image_c = reconstruct_an_image(image_c)
    return image_c


def reconstruct_tensor(input_tensor):
    image_array = []
    with torch.no_grad():
        for images in input_tensor.cpu().numpy():
            image_c = np.array(images)
            image_c = image_c.transpose(1, 2, 0)
            rec = reconstruct_an_image(image_c)
            image_array.append(rec.transpose(2, 0, 1))
        data = np.array(image_array)
    return data


def reconstruct_tensor_rev(input_tensor):
    image_array = []
    with torch.no_grad():
        for img_tensor in input_tensor:
            # Convert tensor to uint8 image
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            # Apply reconstruction steps
            pil_img = Image.fromarray(img_np)
            rec_img = reconstruct_an_image(pil_img)
            # Back to tensor, normalized to [0, 1]
            rec_tensor = torch.from_numpy(rec_img.transpose(2, 0, 1).astype(np.float32)) / 255.0
            rec_tensor = rec_tensor.clamp(0.0, 1.0)  # Ensure valid range
            image_array.append(rec_tensor)
    return torch.stack(image_array)


def partial_reconstruct_tensor_rev_different_psf(input_tensor):
    image_array = []
    with torch.no_grad():
        for images in input_tensor.cpu().numpy():
            image_c = np.array(images)
            image_c = image_c.transpose(1, 2, 0)
            image_c = Image.fromarray((image_c * 255).astype(np.uint8))
            # image_c = make_outer_black(image_c, center_size=(80, 80))
            # gausian_filter = smooth_image_gausian(image_c)
            rec = reconstruct_an_image_different_binary_pattern(image_c)
            image_array.append(rec.transpose(2, 0, 1))
    return torch.from_numpy(np.array(image_array).astype(np.uint8))


def combined_denoise(image, gaussian_radius=1, bilateral_params=(2, 50, 50)):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    smoothed_pil = image.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
    smoothed_cv = np.array(smoothed_pil)
    smoothed_cv = cv2.cvtColor(smoothed_cv, cv2.COLOR_RGB2BGR)
    d, sigma_color, sigma_space = bilateral_params
    denoised_image = cv2.bilateralFilter(smoothed_cv, d, sigma_color, sigma_space)
    return denoised_image


def smooth_image_gausian(image):
    return image.filter(ImageFilter.GaussianBlur(radius=2))


def make_outer_black(image, center_size=(80, 80)):
    image_array = np.array(image, dtype=np.uint8)
    height, width, channels = image_array.shape

    mask = np.zeros((height, width), dtype=np.float32)
    center_h_start = (height - center_size[0]) // 2
    center_w_start = (width - center_size[1]) // 2
    center_h_end = center_h_start + center_size[0]
    center_w_end = center_w_start + center_size[1]

    mask[center_h_start:center_h_end, center_w_start:center_w_end] = 1.0
    mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=10)
    mask = np.expand_dims(mask, axis=-1)

    blended_image = (image_array * mask).astype(np.uint8)
    return Image.fromarray(blended_image, 'RGB')


def visulize_the_psf():
    np.set_printoptions(threshold=np.inf)

    _, _, psf_version_0 = create_mask_for_lensless()
    _, _, psf_version_1 = create_mask_for_lensless()
    _, _, psf_version_2 = create_mask_for_lensless_second_version()

    print(psf_version_0)

    # Plot both PSF images
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(psf_version_1, cmap='gray')
    axes[0].set_title("PSF - Version 1")
    axes[0].axis('off')

    axes[1].imshow(psf_version_2, cmap='gray')
    axes[1].set_title("PSF - Version 2")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate pixel differences between the two PSFs
    diff_psf = np.abs(psf_version_1 - psf_version_2)
    num_different_pixels = np.sum(diff_psf > 1e-6)  # Count the number of pixels where the difference is significant

    print(f"Number of different pixels: {num_different_pixels}")


def convert_into_lensless(input_tensor):
    image_array = []
    with torch.no_grad():
        for images in input_tensor.detach().cpu().numpy():
            image = images.transpose(1, 2, 0)
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            lensless_np = convert_image_to_lensless(pil_image)
            lensless_np = lensless_np.astype(np.float32) / 255.0
            image_array.append(lensless_np.transpose(2, 0, 1))
    return torch.from_numpy(np.array(image_array)).float()


def partial_reconstruct_tensor_rev(input_tensor):
    image_array = []
    with torch.no_grad():
        for img_tensor in input_tensor:
            # Convert tensor to uint8 image
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            # Apply reconstruction steps
            pil_img = Image.fromarray(img_np)
            pil_img = make_outer_black(pil_img, center_size=(64, 64))
            rec_img = reconstruct_an_image(pil_img)
            # Back to tensor, normalized to [0, 1]
            rec_tensor = torch.from_numpy(rec_img.transpose(2, 0, 1).astype(np.float32)) / 255.0
            rec_tensor = rec_tensor.clamp(0.0, 1.0)  # Ensure valid range
            image_array.append(rec_tensor)
    return torch.stack(image_array)


def partial_reconstruct_tensor_rev_two(input_tensor):
    image_array = []
    with torch.no_grad():
        for img_tensor in input_tensor:
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img = make_outer_black(pil_img, center_size=(100, 100))
            smoothed = smooth_image_gausian(pil_img)
            rec_img = reconstruct_an_image(smoothed)
            rec_tensor = torch.from_numpy(rec_img.transpose(2, 0, 1).astype(np.uint8))
            image_array.append(rec_tensor)
    return torch.stack(image_array)
