from dataset.dataset import Dataset
from model.nnc import CNN
from application.train import Training
from application.test import test_model
import torch

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    epochs = 1
    batch_size = 1000
    learning_rate = 1E-6
    best_net: str = ''

    # set device to gpu if available
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # my_device = torch.device('cpu')
    print("using device                                                           ", my_device)

    datasetMNIST = Dataset(batch_size, MNIST=True)

    train_loader_MNIST, val_loader_MNIST, test_loader_MNIST, labels = datasetMNIST.load_dataset()

    model = CNN(class_count=10, device=my_device)
    model.to(my_device)

    training = Training(model, train_loader_MNIST, val_loader_MNIST, epochs, learning_rate, device=my_device)
    flattened_layer, current_predictions = training.train_model()
    
    tsne = TSNE(n_components=2).fit_transform(current_predictions)

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #colors_per_class = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728', 4:'#9467bd', 5:'#8c564b', 6:'#e377c2', 7:'#7f7f7f', 8:'#bcbd22', 9:'#17becf'}
    colors_per_class = {
                        0: (31, 119, 180),
                        1: (255, 127, 14),
                        2: (44, 160, 44),
                        3: (214, 39, 40),
                        4: (148, 103, 189),
                        5: (140, 86, 75),
                        6: (227, 119, 194),
                        7: (127, 127, 127),
                        8: (188, 189, 34),
                        9: (23, 190, 207)
                        }

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = np.array(colors_per_class[label]) / 255
        print("color: ", color)

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()


    #   best_accuracy = test_model(network, test_loader, epochs: int, learning_rate, best_net=None, device: str='cpu')

 #   print("\n MNIST Test accuracy of %f" % (best_accuracy))
#    print(len(flattened_layer), flattened_layer)
    tsne = TSNE(n_components=2, perplexity=50, random_state=0)
#    print('############################################### \n',tsne)
    tsne_output = tsne.fit_transform(flattened_layer)
    

    
#    color_map = plt.cm.get_cmap('gist_rainbow', len(colors))

    plt.scatter(tsne_output[:, 0], tsne_output[:, 1])#, c=colors, cmap=color_map)
    plt.show()


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

if __name__ == "__main__":
    main()
