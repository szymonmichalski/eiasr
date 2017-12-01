import datetime
from collections import Counter

from cv2 import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

from progress.bar import Bar
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

BOW_DICTIONARY_FILENAME = 'bow_dictionary.dat'
SVM_FILENAME = 'svm.dat'
TRAIN_LABELS_FILENAME = 'labels.dat'
TRAIN_DESC_FILENAME = 'descriptors.dat'

TRAIN_DATA_PATH = 'train_data/'
TEST_DATA_PATH = 'train_data/'


class VegeClassifier:
    
    def __init__(self):
        self.train_data_path = ''
        self.test_data_path = ''
        self.class_mapping = {}
        self.class_list = []
        self.train_labels = []
        self.predicted_labels = []
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.sift2 = cv2.xfeatures2d.SIFT_create()
        self.svm = cv2.ml.SVM_create()
        self.BOW = None
        self.bowDiction = None

    def do(self):

        print(datetime.datetime.now())

        self.set_up()

        names_path, training_paths = self.get_training_data()

        self.bag_of_words(training_paths)

        train_labels, train_desc = self.build_labels_array_and_descriptors(
            names_path, training_paths)

        self.train_svm(train_labels, train_desc)

        test_true_labels, test_predicted_labels = self.get_test_labels()

        self.confusion_matrix(test_true_labels, test_predicted_labels)

    def set_up(self):
        self.train_data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            TRAIN_DATA_PATH,
        )
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            TEST_DATA_PATH,
        )
        train_classes = os.listdir(self.train_data_path)
        test_classes = os.listdir(self.test_data_path)
        assert train_classes == test_classes
        self.map_classes(train_classes)
        self.class_list = train_classes
        self.BOW = cv2.BOWKMeansTrainer(len(self.class_mapping))

    def get_training_data(self):
        path = self.train_data_path
        training_names = os.listdir(path)
        training_paths = []
        names_path = []
        for p in training_names:
            pth = os.path.join(path, p)
            training_paths1 = os.listdir(pth)
            for j in training_paths1:
                training_paths.append(os.path.join(pth, j))
                names_path.append(p)
        return names_path, training_paths

    def bag_of_words(self, training_paths):
        if os.path.isfile(BOW_DICTIONARY_FILENAME):
            print('Reading BOW from file')
            with open(BOW_DICTIONARY_FILENAME, 'rb') as f:
                dictionary = np.genfromtxt(f, delimiter=",", dtype=np.float32)
        else:
            print('Adding SIFT descriptors to BOW')
            sift_bar = Bar('SIFT', max=len(training_paths))
            for p in training_paths:
                image = cv2.imread(p)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                kp, dsc = self.sift.detectAndCompute(gray, None)
                self.BOW.add(dsc)
                sift_bar.next()
            sift_bar.finish()
            print(datetime.datetime.now())
            print('Building BOW cluster')
            dictionary = self.BOW.cluster()
            print('Saving BOW dictionary to file')
            with open(BOW_DICTIONARY_FILENAME, 'wb') as f:
                np.savetxt(f, dictionary, delimiter=",")

        print(datetime.datetime.now())
        print('Building BOW dictionary')
        self.bowDiction = cv2.BOWImgDescriptorExtractor(
            self.sift2,
            cv2.BFMatcher(cv2.NORM_L2)
        )
        self.bowDiction.setVocabulary(dictionary)
        print("BOW dictionary: ", np.shape(dictionary))

    def build_labels_array_and_descriptors(self, names_path, training_paths):
        if (
                os.path.isfile(TRAIN_LABELS_FILENAME) and
                os.path.isfile(TRAIN_DESC_FILENAME)
        ):
            print('Reading train labels from file')
            with open(TRAIN_LABELS_FILENAME, 'rb') as f:
                train_labels = np.genfromtxt(f, delimiter=",", dtype=np.int32)
            print('Reading train descriptors from file')
            with open(TRAIN_DESC_FILENAME, 'rb') as f:
                train_desc = np.genfromtxt(f, delimiter=",", dtype=np.int32)
        else:
            print(datetime.datetime.now())
            print('Building labels array')
            train_desc = []
            train_labels = []
            i = 0
            label_bar = Bar('Labels', max=len(training_paths))
            for p in training_paths:
                train_desc.extend(self.feature_extract(p))
                train_labels.append(self.class_mapping[names_path[i]])
                i = i + 1
                label_bar.next()
            label_bar.finish()
            # print(Counter(train_labels))
            with open(TRAIN_LABELS_FILENAME, 'wb') as f:
                np.savetxt(f, train_labels, delimiter=",")
            with open(TRAIN_DESC_FILENAME, 'wb') as f:
                np.savetxt(f, train_desc, delimiter=",")
        return train_labels, train_desc

    def train_svm(self, train_labels, train_desc):
        if os.path.isfile(SVM_FILENAME):
            print('Reading SVM from file')
            self.svm = cv2.ml.SVM_load(SVM_FILENAME)
        else:
            print(datetime.datetime.now())
            print('Training SVM')
            print("svm items", len(train_desc), len(train_desc[0]))
            x = np.array(train_desc, dtype=np.float32)
            y = np.array(train_labels, dtype=np.int32)
            self.svm.setKernel(cv2.ml.SVM_LINEAR)
            self.svm.trainAuto(x, cv2.ml.ROW_SAMPLE, y)
            self.svm.save(SVM_FILENAME)

    def classify_images(self, training_paths):
        print(datetime.datetime.now())
        print('Calculating confusion matrix')
        classify_bar = Bar('Classify', max=len(training_paths))
        for img_pth in training_paths:
            self.predicted_labels.append(self.classify(img_pth))
            classify_bar.next()
        classify_bar.finish()

    def classify(self, img_pth):
        feature = self.feature_extract(img_pth)
        p = self.svm.predict(feature)
        return int(p[1].item((0, 0)))

    def feature_extract(self, img_pth):
        img = cv2.imread(img_pth, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.bowDiction.compute(gray, self.sift.detect(gray))

    def get_test_labels(self):
        print('Getting labels')
        true_labels = []
        predicted_labels = []
        path = self.test_data_path
        class_folders = os.listdir(path)
        for cls in class_folders:
            pth = os.path.join(path, cls)
            training_paths1 = os.listdir(pth)
            for j in training_paths1:
                predicted_labels.append(self.classify(os.path.join(pth, j)))
                true_labels.append(self.class_mapping[cls])

        return (
            self.change_labels(true_labels),
            self.change_labels(predicted_labels)
        )

    def map_classes(self, train_classes):
        i = 0
        for cls in train_classes:
            self.class_mapping[cls] = i
            i += 1

    def change_labels(self, labels):
        inverted_classes = {v: k for k, v in self.class_mapping.items()}
        changed_labels = []
        for l in labels:
            changed_labels.append(inverted_classes[l])
        return changed_labels

    def confusion_matrix(self, test_true_labels, test_predicted_labels):
        cnf_matrix = confusion_matrix(
            test_true_labels, test_predicted_labels, labels=self.class_list)
        print('True: {}'.format(Counter(test_true_labels)))
        print('Predicted: {}'.format(Counter(test_predicted_labels)))
        print(self.class_list)
        print(cnf_matrix)
        print(precision_recall_fscore_support(
            test_true_labels, test_predicted_labels, average='weighted'))
        print(precision_recall_fscore_support(
            test_true_labels, test_predicted_labels, average='micro'))
        print(precision_recall_fscore_support(
            test_true_labels, test_predicted_labels, average='macro'))

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(
            cnf_matrix,
            classes=self.class_list,
            title='Confusion matrix, without normalization',
        )

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(
            cnf_matrix,
            classes=self.class_list,
            normalize=True,
            title='Normalized confusion matrix',
        )
        plt.show()


def plot_confusion_matrix(
        cm, classes,
       normalize=False,
       title='Confusion matrix',
       cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    classifier = VegeClassifier()
    classifier.do()
