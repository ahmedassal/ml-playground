from utils import get_mnist, get_gabors
import matplotlib.pyplot as plt
import numpy as np
import mdp
from sklearn.decomposition import sparse_encode
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy import signal
from scipy import ndimage
import cv2


from skimage.feature import hog
from skimage import data, color, exposure


scaler = MinMaxScaler(feature_range=(0, 1))


class SpatialCoder(object):

    activation_threshold = 0.1
    zero_value = -10
    orientation_threshold = 2

    def __init__(self, img, patch_size=5):
        self.img = img
        # self.img = ndimage.gaussian_filter(img, .5) - ndimage.gaussian_filter(img, 1)
        self.kernels = get_gabors(patch_size)
        self.img_height, self.img_width = img.shape
        self.patch_size = patch_size
        self.map_height, self.map_width = (
            self.img_height - patch_size + 1,
            self.img_width - patch_size + 1
        )

    def draw_kernels(self, x=7, y=5):
        for i in xrange(self.kernels.shape[0]):
            img = self.kernels[i].reshape(
                self.patch_size, self.patch_size
            )
            plt.subplot(x, y, i)
            plt.xticks(())
            plt.yticks(())
            plt.imshow(img, cmap='gray')
        plt.show()

    def hog_encode(self):
        fd, hog_image = hog(
            self.img, orientations=8, pixels_per_cell=(5, 5),
            cells_per_block=(5, 5), visualise=True
        )
        return hog_image

    def gabor_encode(self):
        patches = extract_patches_2d(
            self.img, (self.patch_size, self.patch_size)
        )
        patches = patches.reshape(patches.shape[0], -1)
        # code = sparse_encode(patches, self.kernels, algorithm='threshold', alpha=1)
        code = sparse_encode(
            patches, self.kernels, algorithm='lars', n_nonzero_coefs=2)

        idx = np.std(code, axis=1) > 0.3
        selected_patches = patches #[idx]
        selected_code = code #[idx]
        min_code, max_code = np.min(selected_code), np.max(selected_code)
        # print selected_patches
        c = 0
        s = 21
        for i in xrange(selected_code.shape[0]):
            print i

            plt.subplot(s, s * 2, c)
            plt.xticks(())
            plt.gca().set_ylim([min_code, max_code])
            plt.yticks(())
            plt.plot(selected_code[i])
            c += 1

            plt.subplot(s, s * 2, c)
            plt.xticks(())
            plt.yticks(())
            plt.imshow(selected_patches[i].reshape(
                self.patch_size, self.patch_size), cmap='gray', interpolation='none')
            c += 1
        plt.show()

        orientations = np.argmax(code, axis=1)
        activations = np.std(code, axis=1)
        orientations[activations < self.activation_threshold] = self.zero_value
        # blank_batches = np.ones((patches.shape[0], self.patch_size, self.patch_size)) * orientations[:, None, None]
        # recon = reconstruct_from_patches_2d(blank_batches, (self.img_height, self.img_width))
        # return recon
        return orientations.reshape(self.map_height, self.map_width)

    def gabor_convolve(self):
        convs = []
        for k in self.kernels:
            k = k.reshape(self.patch_size, self.patch_size)
            convs.append(signal.convolve2d(self.img, k, mode='valid'))
        convs = np.dstack(convs)
        # convs = np.abs(convs)
        code = np.zeros(
            (self.map_height * self.map_width, self.kernels.shape[0])
        )
        # plt.imshow(self.img, cmap='gray', interpolation='none')
        # plt.show()
        print self.kernels.shape
        for i in xrange(self.kernels.shape[0]):
            plt.subplot(3, 6, i)
            plt.xticks(())
            plt.yticks(())
            plt.imshow(convs[:, :, i], cmap='gray', interpolation='none')
            # print self.kernels[i].reshape(self.patch_size, self.patch_size)
            # raw_input()
            # plt.imshow(self.kernels[i].reshape(self.patch_size, self.patch_size), cmap='gray', interpolation='none')
        plt.show()

        patches = np.zeros(
            (self.map_height * self.map_width, self.patch_size * self.patch_size))
        for i in xrange(self.map_height):
            for j in xrange(self.map_width):
                idx = i * self.map_height + j
                patches[idx] = self.img[
                    i: i + self.patch_size, j: j + self.patch_size].ravel()
                code[idx] = convs[i, j]

        code = np.abs(code)
        print code.shape
        idx = np.std(code, axis=1) > 0.3
        selected_patches = patches #[idx]
        selected_code = code #[idx]
        min_code, max_code = np.min(selected_code), np.max(selected_code)
        # print selected_patches
        c = 0
        s = self.map_height
        for i in xrange(selected_code.shape[0]):
            print i

            p = selected_patches[i].reshape(
                self.patch_size, self.patch_size)

            plt.subplot(s, s * 2, c)
            plt.xticks(())
            plt.gca().set_ylim([min_code, max_code])
            plt.yticks(())
            plt.plot(selected_code[i])
            c += 1

            plt.subplot(s, s * 2, c)
            plt.xticks(())
            plt.yticks(())
            plt.imshow(p, cmap='gray', interpolation='none')
            c += 1
        plt.show()

        orientations = np.argmax(code, axis=1)
        activations = np.std(code, axis=1)
        orientations[activations < self.activation_threshold] = self.zero_value
        # # blank_batches = np.ones((patches.shape[0], self.patch_size, self.patch_size)) * orientations[:, None, None]
        # # recon = reconstruct_from_patches_2d(blank_batches, (self.img_height, self.img_width))
        # # return recon
        return orientations.reshape(self.map_height, self.map_width)

    def distances_for_lvl1(self, data):
        m, n = data.shape
        # initialize with something big
        dist = np.ones((m, m)) * (self.map_width * 2)
        for i in xrange(m):
            for j in xrange(m):
                if i == j:
                    dist[i, j] = 0
                else:
                    a = data[i]
                    b = data[j]
                    if a[2] == self.zero_value or b[2] == self.zero_value:
                        continue
                    delta = np.abs(a[2] - b[2])
                    if delta > self.orientation_threshold:
                        continue
                    dist[i, j] = np.linalg.norm(a[:2] - b[:2])
                    # print dist[i, j]
        return dist

    def cluster_lvl1(self, data):
        db = DBSCAN(eps=2., min_samples=2, metric='precomputed')
        processed = np.float32(np.vstack([
            np.mgrid[:self.map_height, :self.map_width].reshape(2, -1),
            data.ravel()
        ])).T
        dist = self.distances_for_lvl1(processed)
        return db.fit_predict(dist).reshape(self.map_height, self.map_width)


if __name__ == '__main__':

    # data, target = get_mnist(20000, 20064)
    # data = (np.float32(data) / 255.).reshape(64, 28, 28)
    # data -= np.mean(data)
    # for i in xrange(64):
    #     print i
    #     coder = SpatialCoder(data[i])
    #     # result = coder.gabor_encode()
    #     result = coder.gabor_convolve()
    #     # result = coder.hog_encode()
    #     result = coder.cluster_lvl1(result)
    #     plt.subplot(8, 8, i)
    #     plt.xticks(())
    #     plt.yticks(())
    #     plt.imshow(result, interpolation='none')
    # plt.show()

    data, target = get_mnist(20000, 20001)
    data = (np.float32(data) / 255.).reshape(28, 28)
    data -= np.mean(data)
    coder = SpatialCoder(data)
    omap = coder.gabor_convolve()
    # result = coder.cluster_lvl1(omap)

    # plt.subplot(2, 1, 0)
    # plt.xticks(())
    # plt.yticks(())
    plt.imshow(omap, interpolation='none')
    # plt.subplot(2, 1, 1)
    # plt.xticks(())
    # plt.yticks(())
    # plt.imshow(result, interpolation='none')
    plt.show()
