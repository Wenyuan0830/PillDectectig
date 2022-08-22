import numpy as np


def generate_gaussian_kernel(sigma):
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    kernel = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return kernel


class StackedHeatmapGenerator:
    """ Use this to generate gaussian kernels on a single heatmap.

    Note this is different than the commonly used protocol in 2D keypoint detection task.

    Reference:
        https://github.com/greatlog/SWAHR-HumanPose/blob/master/lib/dataset/target_generators/target_generators.py#L15
    """

    def __init__(self, output_res, n_hms=2, rescale_factor=1., sigma=-1):
        self.output_res = output_res
        self.n_hms = n_hms  # missing and present
        self.rescale_factor = rescale_factor
        if sigma < 0:
            sigma = self.output_res / 64
        self.sigma = sigma
        self.g = generate_gaussian_kernel(self.sigma)

    def __call__(self, joints):
        sigma = self.sigma

        hms = np.zeros(
            (
                self.n_hms,
                self.output_res,
                self.output_res
            ),
            dtype=np.float32
        )

        for p in joints:
            x, y, cls = int(p[0] * self.rescale_factor), int(p[1] * self.rescale_factor), p[2]

            if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                continue

            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.output_res)
            aa, bb = max(0, ul[1]), min(br[1], self.output_res)
            hms[cls, aa:bb, cc:dd] = np.maximum(hms[cls, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    hg = StackedHeatmapGenerator(256)

    joint_array = np.asarray([
        [80, 150, 0],
        [200, 200, 0],
        [100, 160, 1],
        [175, 40, 1]
    ])

    hms = hg(joint_array)

    for item in hms:
        plt.figure()
        plt.imshow(item)

    plt.show()
