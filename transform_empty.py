from image import Image
import numpy as np


def brighten(image: Image, factor=2.0) -> Image:
    # when we brighten, we just want to make each channel higher by some amount
    # factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)

    # make empty image so we don't modify original one
    new_img = Image(image.array.shape)

    # non-vectorised way (easier to understand):
    # brighten image by multiplying each pixel by the factor

    # for x in range(x_pixels):
    #     for y in range(y_pixels):
    #         for c in range(num_channels):
    #             new_img.array[x, y, c] = image.array[x, y, c] * factor

    # vectorised version which leverages numpy (does the same as the above, but faster):
    new_img.array = image.array * factor

    return new_img


def adjust_contrast(image: Image, factor: float, mid=0.5) -> Image:
    # adjust the contrast by increasing the difference from the user-defined midpoint by
    # factor amount

    new_img = Image(image.array.shape)

    # for x in range(new_img.x_pixels):
    #     for y in range(new_img.y_pixels):
    #         for c in range(new_img.num_channels):
    #             new_img.array[x, y, c] = ((image.array[x, y, c] - mid) * factor) + mid

    # vectorised:
    new_img.array = ((image.array - mid) * factor) + mid

    return new_img


def blur(image: Image, kernel_size: int) -> Image:
    # we blur by averaging a pixel with its neighbours (which will be determined by the kernel)
    # kernel size is the number of pixels to take into account when applying the blur
    # (ie kernel_size = 3 would be kernel_size / 2 neighbours to the left/right, top/bottom,
    # and diagonals)
    # kernel size should always be an *odd* number

    new_img = Image(image.array.shape)

    # this is a variable that tells us how many neighbours we actually look at (ie for a kernel of
    # 3, this value should be 1)
    # how many neighbours to one side we need to look at
    neighbour_range = kernel_size // 2

    for x in range(new_img.x_pixels):
        for y in range(new_img.y_pixels):
            for c in range(new_img.num_channels):
                total = 0

                for x_i in range(
                    max(0, x - neighbour_range),
                    min(new_img.x_pixels - 1, x + neighbour_range) + 1,
                ):
                    for y_i in range(
                        max(0, y - neighbour_range),
                        min(new_img.y_pixels - 1, y + neighbour_range) + 1,
                    ):
                        total += image.array[x_i, y_i, c]

                # kernel_size**2 because the neighbours create a perfect square
                # this is the total number of neighbours to consider
                new_img.array[x, y, c] = total / (kernel_size**2)

    return new_img

    # note:
    # the blur implemented above is a kernel of size n, where each value is 1/n^2
    # for example, k=3 would be this kernel:
    # [1/3 1/3 1/3]
    # [1/3 1/3 1/3]
    # [1/3 1/3 1/3]


def apply_kernel(image: Image, kernel: np.ndarray) -> Image:
    # 'kernel' should be a 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]

    new_img = Image(image.array.shape)
    kernel_size = kernel.shape[0]
    neighbour_range = kernel_size // 2

    for x in range(new_img.x_pixels):
        for y in range(new_img.y_pixels):
            for c in range(new_img.num_channels):
                total = 0

                for x_i in range(
                    max(0, x - neighbour_range),
                    min(new_img.x_pixels - 1, x + neighbour_range) + 1,
                ):
                    for y_i in range(
                        max(0, y - neighbour_range),
                        min(new_img.y_pixels - 1, y + neighbour_range) + 1,
                    ):
                        x_k = x_i + neighbour_range - x
                        y_k = y_i + neighbour_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_img.array[x, y, c] = total

    return new_img


def combine_images(image1: Image, image2: Image) -> Image:
    # let's combine two images using the squared sum of squares: value = sqrt(value_1**2, value_2**2)
    # size of image1 and image2 MUST be the same

    new_im = Image(image1.array.shape)

    for x in range(new_im.x_pixels):
        for y in range(new_im.y_pixels):
            for c in range(new_im.num_channels):
                new_im.array[x, y, c] = (
                    image1.array[x, y, c] ** 2 + image2.array[x, y, c] ** 2
                ) ** 0.5
    return new_im


if __name__ == "__main__":
    print("Processing image. Hold on...")

    lake = Image(filename="lake.png")
    city = Image(filename="city.png")

    # brightened image
    # brighten(lake).write_image("bright_lake.png")

    # reduce brightness (darken)
    # brighten(lake, 0.5).write_image("dark_lake.png")

    # increase contrast
    # adjust_contrast(lake, 2.0).write_image("increased_contrast_lake.png")

    # decrease contrast
    # adjust_contrast(lake, 0.5).write_image("decreased_contrast_lake.png")

    # blurred with kernel of 3
    # blur(city, 3).write_image("blurred_city_kernel_3.png")

    # blurred with kernel of 15
    # blur(city, 15).write_image("blurred_city_kernel_15.png")

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

    # combine_images(
    #     Image(filename="edge_x.png"), Image(filename="edge_y.png")
    # ).write_image("edge_xy.png")

    print(
        f"Done processing image. Find the new image in the {lake.output_path} folder."
    )
