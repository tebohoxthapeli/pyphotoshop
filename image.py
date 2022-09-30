from numpy import zeros, clip, vstack
from png import Reader, Writer


class Image:
    # def __init__(self, x_pixels=0, y_pixels=0, num_channels=0, filename=""):
    #     # you need to input either filename OR x_pixels, y_pixels, and num_channels
    #     self.input_path = "input/"
    #     self.output_path = "output/"

    #     if x_pixels and y_pixels and num_channels:
    #         self.x_pixels = x_pixels
    #         self.y_pixels = y_pixels
    #         self.num_channels = num_channels
    #         self.array = zeros((x_pixels, y_pixels, num_channels))
    #     elif filename:
    #         self.array = self.read_image(filename)
    #         self.x_pixels, self.y_pixels, self.num_channels = self.array.shape
    #     else:
    #         raise ValueError(
    #             "You need to input either a filename OR specify the dimensions of the image"
    #         )

    def __init__(
        self, shape: tuple[int, int, int] | None = None, filename: str | None = None
    ):
        # you need to input either filename OR x_pixels, y_pixels, and num_channels

        self.input_path = "input/"
        self.output_path = "output/"

        if shape:
            self.x_pixels = shape[0]
            self.y_pixels = shape[1]
            self.num_channels = shape[2]
            self.array = zeros(shape)

        elif filename:
            self.array = self.read_image(filename)
            self.x_pixels, self.y_pixels, self.num_channels = self.array.shape

        else:
            raise ValueError(
                "You need to input either a filename OR specify the dimensions of the image"
            )

    def read_image(self, filename, gamma=2.2):
        """
        read PNG RGB image, return 3D numpy array organized along Y, X, channel
        values are float, gamma is decoded
        """
        im = Reader(self.input_path + filename).asFloat()
        resized_image = vstack(list(im[2]))
        resized_image.resize(im[1], im[0], 3)
        resized_image = resized_image**gamma
        return resized_image

    def write_image(self, output_file_name: str, gamma=2.2):
        """
        3D numpy array (Y, X, channel) of values between 0 and 1 -> write to png
        """
        im = clip(self.array, 0, 1)
        y, x = self.array.shape[0], self.array.shape[1]
        im = im.reshape(y, x * 3)
        writer = Writer(x, y)
        with open(self.output_path + output_file_name, "wb") as f:
            writer.write(f, 255 * (im ** (1 / gamma)))

        self.array.resize(
            y, x, 3
        )  # we mutated the method in the first step of the function


if __name__ == "__main__":
    im = Image(filename="lake.png")
    im.write_image("test.png")
