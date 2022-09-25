from image import Image
import numpy as np


def get_image_copy(old_img_arr_shape: tuple[int, ...]) -> Image:
    x_pixel, y_pixel, num_channels = old_img_arr_shape
    return Image(x_pixel, y_pixel, num_channels)


def brighten(image: Image, factor: float) -> Image:
    # when we brighten, we just want to make each channel higher by some amount
    # factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)

    # make empty image so we don't modify original one
    new_im = get_image_copy(image.array.shape)

    # this is the most intuitive way to do this (non-vectorised)
    # for x in range(x_pixels):
    #     for y in range(y_pixels):
    #         for c in range(num_channels):
    #             new_im.array[x, y, c] = image.array[x, y, c] * factor

    # vectorised version (does the same as the above, but faster)
    new_im.array = image.array * factor

    return new_im


def adjust_contrast(image: Image, factor: float, mid: float = 0.5) -> Image:
    # adjust the contrast by increasing the difference from the user-defined midpoint by factor amount
    new_im = get_image_copy(image.array.shape)

    for x in range(new_im.x_pixels):
        for y in range(new_im.y_pixels):
            for c in range(new_im.num_channels):
                new_im.array[x, y, c] = (image.array[x, y, c] - mid) * factor + mid

    # vectorised
    # new_im.array = (image.array - mid) * factor + mid

    return new_im


def blur(image: Image, kernel_size: int) -> Image:
    # kernel size is the number of pixels to take into account when applying the blur
    # (ie kernel_size = 3 would be neighbors to the left/right, top/bottom, and diagonals)
    # kernel size should always be an *odd* number

    new_im = get_image_copy(image.array.shape)
    neighbour_range = kernel_size // 2

    for x in range(new_im.x_pixels):
        for y in range(new_im.y_pixels):
            for c in range(new_im.num_channels):
                total = 0

                for x_i in range(
                    max(0, x - neighbour_range),
                    min(new_im.x_pixels - 1, x + neighbour_range) + 1,
                ):
                    for y_i in range(
                        max(0, y - neighbour_range),
                        min(new_im.y_pixels - 1, y + neighbour_range) + 1,
                    ):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size**2)

    return new_im

    # note:
    # this blur implemented above is a kernel of size n, where each value if 1/n^2
    # for example, k=3 would be this kernel:
    # [1/3 1/3 1/3]
    # [1/3 1/3 1/3]
    # [1/3 1/3 1/3]


def apply_kernel(image: Image, kernel: np.ndarray) -> Image:
    # the kernel should be a 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]

    new_im = get_image_copy(image.array.shape)
    kernel_size = kernel.shape[0]
    neighbour_range = kernel_size // 2

    for x in range(new_im.x_pixels):
        for y in range(new_im.y_pixels):
            for c in range(new_im.num_channels):
                total = 0

                for x_i in range(
                    max(0, x - neighbour_range),
                    min(new_im.x_pixels - 1, x + neighbour_range) + 1,
                ):
                    for y_i in range(
                        max(0, y - neighbour_range),
                        min(new_im.y_pixels - 1, y + neighbour_range) + 1,
                    ):
                        x_k = x_i + neighbour_range - x
                        y_k = y_i + neighbour_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_im.array[x, y, c] = total

    return new_im


def combine_images(image1, image2):
    # let's combine two images using the squared sum of squares: value = sqrt(value_1**2, value_2**2)
    # size of image1 and image2 MUST be the same
    pass


if __name__ == "__main__":
    lake = Image(filename="lake.png")
    city = Image(filename="city.png")

    # brightened_im = brighten(lake, 2.0)
    # brightened_im.write_image("brightened_image.png")

    # reduced brightness
    # brighten(lake, 0.5).write_image("reduced_brightness_lake.png")

    # increase contrast
    # adjust_contrast(lake, 2.0).write_image("increased_contrast_lake.png")

    # decrease contrast
    # adjust_contrast(lake, 0.5).write_image("decreased_contrast_lake.png")

    # blurred with kernel of 3
    # blur(lake, 3).write_image("blurred_lake_3.png")

    # blurred with kernel of 15
    # blur(lake, 15).write_image("blurred_lake_15.png")

    # apply sobel edge detection kernel on the x and y axis

    sobel_x_kernel = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]
    )

    sobel_y_kernel = np.array(
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]
    )

    apply_kernel(city, sobel_x_kernel).write_image("edge_x.png")
    apply_kernel(city, sobel_y_kernel).write_image("edge_y.png")

    print("Done processing image.")
