import datetime
import numpy as np
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold
import matplotlib.pyplot as plt
import pandas as pd


def get_current_time():
    currentDT = datetime.datetime.now()
    return str(currentDT.strftime("%Y-%m-%d-%H-%M-%S"))


def imscatter(x, y, ax, imageData, zoom, imageSize):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        # img = img.astype(np.uint8).reshape([imageSize,imageSize])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))


# Show dataset images with T-sne projection of latent space encoding
def compute_TSNE_projection_of_latent_space(z_s, y_s, display=True):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(z_s)
    print(X_tsne)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(frameon=False)
        plt.title("T-SNE")
        plt.setp(ax, xticks=(), yticks=())
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                        wspace=0.0, hspace=0.0)

        color_map = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple', 5: 'brown', 6: 'pink', 7: 'gray',
                     8: 'olive', 9: 'cyan'}
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_s, marker="x")
        plt.colorbar()
        plt.savefig('tsne.png', dpi=300)
        plt.show()

    else:
        return X_tsne




