import numpy as np
# load test set for evaluation
test_df = pd.read_csv('loan_test.csv')
test_df.head()


# from sklearn.metrics import classification_report, confusion_matrix
# import itertools
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
# np.set_printoptions(precision=2)
#
# print (classification_report(y_test, yhat))
#
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


# NEEDED!!!

X_test = np.asarray(test_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X_test[0:5])
y_test = np.asarray(test_df['churn'])
print(y_test [0:5])
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
print ('Test set:', X_test.shape,  y_test.shape)


from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat))

# LOG LOSS
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)