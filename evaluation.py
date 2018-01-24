import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import interp

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import MultiLabelBinarizer


class Evaluation:

    def __init__(self, test_true_labels, test_predicted_labels, class_list):
        self.test_true_labels = test_true_labels
        self.test_predicted_labels = test_predicted_labels
        self.class_list = class_list
        self.cnf_matrix = confusion_matrix(
            test_true_labels, test_predicted_labels, labels=class_list)
        self.acc_all = precision_recall_fscore_support(
            test_true_labels, test_predicted_labels)
        self.acc_weight = precision_recall_fscore_support(
            test_true_labels, test_predicted_labels, average='weighted')
        self.acc_micro = precision_recall_fscore_support(
            test_true_labels, test_predicted_labels, average='micro')
        self.acc_macro = precision_recall_fscore_support(
            test_true_labels, test_predicted_labels, average='macro')

    def plot_cnf_matrix(self):
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(
            self.cnf_matrix,
            classes=self.class_list,
            title='Confusion matrix, without normalization',
        )

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(
            self.cnf_matrix,
            classes=self.class_list,
            normalize=True,
            title='Normalized confusion matrix',
        )
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
            cm, classes, normalize=False, title='Confusion matrix',
            cmap=plt.cm.Blues,
    ):
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

    def plot_roc(self, lw=2):
        # Compute ROC curve and ROC area for each class
        n_classes = len(self.class_list)
        y_test, y_score, cls_mapping = self.cls_to_binary(
            self.test_true_labels, self.test_predicted_labels
        )
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # print(y_score)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(
            fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'.format(
                roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

        plt.plot(
            fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'.format(
                roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'.format(
                    cls_mapping[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating '
                  'characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    def cls_to_binary(self, true, predicted):
        cls_true = []
        cls_pred = []
        class_mapping = {}
        for i in range(len(self.class_list)):
            class_mapping[i] = self.class_list[i]
            for j in range(len(true)):
                if true[j] == self.class_list[i]:
                    cls_true.append([self.class_list.index(true[j])])
                    cls_pred.append([self.class_list.index(predicted[j])])
        return (
            MultiLabelBinarizer().fit_transform(cls_true),
            MultiLabelBinarizer().fit_transform(cls_pred),
            class_mapping
        )
