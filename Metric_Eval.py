import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.datasets import cifar10, cifar100
from keras.models import load_model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import keras
import matplotlib as mpl

mpl.use('Agg')
def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def print_metrics(y_true, y_pred, dataset_type='Test'):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print(f"{dataset_type} Accuracy: {accuracy:.4f}")
    print(f"{dataset_type} Precision: {precision:.4f}")
    print(f"{dataset_type} Recall: {recall:.4f}")
    print(f"{dataset_type} F1-Score: {f1_score:.4f}")


def plot_and_save_confusion_matrix(y_true, y_pred, path, dataset_type='Test'):

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{dataset_type} Confusion Matrix')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# load data
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)
#%%
model = load_model('/data0/jinhaibo/Remote/STAT/models/densenet121-dataaug.h5')

y_train_pred_probs = model.predict(x_train)
y_test_pred_probs = model.predict(x_test)

y_train_pred = np.argmax(y_train_pred_probs, axis=1)
y_test_pred = np.argmax(y_test_pred_probs, axis=1)

y_train_true = np.argmax(y_train, axis=1)
y_test_true = np.argmax(y_test, axis=1)

print_metrics(y_train_true, y_train_pred, 'Train')
print_metrics(y_test_true, y_test_pred, 'Test')

plot_and_save_confusion_matrix(y_train_true, y_train_pred, '/data0/jinhaibo/Remote/STAT/Figs/train_confusion_matrix_dense_aug.png', 'Train')
plot_and_save_confusion_matrix(y_test_true, y_test_pred, '/data0/jinhaibo/Remote/STAT/Figs/test_confusion_matrix_dense_aug.png', 'Test')