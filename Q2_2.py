import numpy as np
from cvxopt import matrix, solvers
import os
import cv2
from sklearn.metrics import accuracy_score

def combinations_of_2(classes):
    n =  classes.size
    comb = []
    if n == 0 or n == 1:
        return np.array([]) 
    else:
        for i in range(n):
            j = i + 1
            while(j < n):
                comb.append((classes[i], classes[j]))
                j += 1
        return np.array(comb)

############### 2.2 Part-(a) ################

class SVM_multiclass_classifier:
    def __init__(self, path, C=1.0, gamma=0.001):
        self.path = path
        self.C = C
        self.gamma = gamma
        self.classifiers = []
    
    def flatten_images(self, data):
        flattened_images = data.reshape(data.shape[0], -1) / 255.0
        return flattened_images
    
    def resize_data(self, Y, validation = False):
        self.classes = np.unique(Y)
        num_classes = len(self.classes)
        images = []
        for i in range(num_classes):
            if validation:
                img_folder = os.path.join(self.path, f"{i}")
            else:
                img_folder = os.path.join(self.path, f"{i}(1)")
            for img in os.listdir(img_folder):
                path = os.path.join(img_folder, img)
                img_data = cv2.imread(path)
                rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(rgb_image, (16, 16))
                images.append(resized_image)
        return np.array(images)

    def one_vs_one_svm(self, X, Y):
        # Generate all possible pairs of classes
        class_pairs = combinations_of_2(self.classes)

        for class_1, class_2 in class_pairs:
            # Select data for the current pair of classes
            X_pair = []
            Y_pair = []
            for i in range(len(Y)):
                if Y[i] == class_1 or Y[i] == class_2:
                    X_pair.append(X[i])
                    Y_pair.append(Y[i])
            X_pair = np.array(X_pair)
            Y_pair = np.array(Y_pair)
            Y_pair_1 = np.where(Y_pair == class_1, 1, -1)  # Convert to binary labels
            
            # Train a binary SVM classifier for this pair of classes
            alphas = self.train_binary_svm(X_pair, Y_pair_1)
            supp_vec_ind = np.where(alphas > 1e-4)[0]
            num_supp_vecs = len(supp_vec_ind)
            vec = np.zeros(num_supp_vecs)
            support_vectors = X_pair[supp_vec_ind]
            support_vector_labels = Y_pair_1[supp_vec_ind]
            for supp_vec_ind in np.where(alphas > 1e-4)[0]:
                vec = vec + Y_pair_1[supp_vec_ind]*alphas[supp_vec_ind]*np.exp(-self.gamma* np.sum((support_vectors - X_pair[supp_vec_ind])*(support_vectors - X_pair[supp_vec_ind]), axis = 1))
            b = np.mean(support_vector_labels - vec)
            
            self.classifiers.append((class_1, class_2, alphas, X_pair, Y_pair_1, b))

    def train_binary_svm(self, X, Y):
        self.m = X.shape[0]
        self.n = X.shape[1]
        P = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                P[i, j] = Y[i] * Y[j] * np.exp(-self.gamma * np.linalg.norm(X[i] - X[j]) ** 2)
        q = matrix(-np.ones(self.m))
        G = matrix(np.vstack((-np.eye(self.m), np.eye(self.m))))
        h = matrix(np.hstack((np.zeros(self.m), self.C * np.ones(self.m))))
        A = matrix(Y.reshape(1, -1), (1, 4760), 'd')
        b = matrix(np.zeros(1))
        sol = solvers.qp(matrix(P), q, G, h, A, b)
        alphas = np.array(sol['x']).flatten()

        return alphas
    
    def predict(self, X_validation):
        num_samples = X_validation.shape[0]
        votes = np.zeros((num_samples, len(self.classes)))

        for _, row in enumerate(self.classifiers):
            class_1 = row[0]
            class_2 = row[1]
            alphas = row[2]
            X_pair = row[3]
            Y_pair = row[4]
            b = row[5]
            for i in range(num_samples):
                prediction = b
                for j in range(len(alphas)):
                    prediction += alphas[j] * Y_pair[j] * np.exp(-self.gamma * (np.linalg.norm(X_pair[j] - X_validation[i]) ** 2))
                    
                if prediction > 0:
                    pred_class = class_1
                else:
                    pred_class = class_2
                votes[i, pred_class] += 1
        #print(votes)
        Y_pred = np.argmax(votes, axis=1)
        return Y_pred

# Loading the training and the validation datasets.
folder_path = 'C:\\Users\\Acer\\Desktop\\IITD\\3rd year\\Sem. 1\\COL774\\a2\\Q2'

# Create and train the multiclass SVM model
multiclass_classifier = SVM_multiclass_classifier(folder_path, C=1.0, gamma=0.001)
Y_train = []
for i in range(6):
    img_folder = os.path.join(folder_path, f"{i}(1)")
    for img in os.listdir(img_folder):
        Y_train.append(i)
Y_train = np.array(Y_train)
#print(Y_train)
X_train_resized = multiclass_classifier.resize_data(Y_train)
X_train = multiclass_classifier.flatten_images(X_train_resized)
Y_validation = []
for i in range(6):
    img_folder = os.path.join(folder_path, f"{i}")
    for img in os.listdir(img_folder):
        Y_validation.append(i)
Y_validation = np.array(Y_validation)
X_validation_resized = multiclass_classifier.resize_data(Y_validation, True)
X_validation = multiclass_classifier.flatten_images(X_validation_resized)
#print(X_train)
multiclass_classifier.one_vs_one_svm(X_train, Y_train)

# Predict the validation set
Y_pred_a = multiclass_classifier.predict(X_validation)

# Calculate accuracy
accuracy = 0
n1 = len(Y_pred_a)
for i in range(n1):
    if Y_pred_a[i] == Y_validation[i]:
        accuracy += 1/n1
print("Validation set accuracy:", accuracy * 100)


################### 2.2 Part-(b) #########################
import numpy as np
import os
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

def flatten_images(data):
    flattened_images = data.reshape(data.shape[0], -1) / 255.0
    return flattened_images
    
def resize_data(Y, path, validation = False):
    classes = np.unique(Y)
    num_classes = len(classes)
    images = []
    for i in range(num_classes):
        if validation:
            img_folder = os.path.join(path, f"{i}")
        else:
            img_folder = os.path.join(path, f"{i}(1)")
        for img in os.listdir(img_folder):
            path1 = os.path.join(img_folder, img)
            img_data = cv2.imread(path1)
            rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(rgb_image, (16, 16))
            images.append(resized_image)
    return np.array(images)
folder_path = 'C:\\Users\\Acer\\Desktop\\IITD\\3rd year\\Sem. 1\\COL774\\a2\\Q2'

# Create and train the multiclass SVM model
Y_train = []
for i in range(6):
    img_folder = os.path.join(folder_path, f"{i}(1)")
    for img in os.listdir(img_folder):
        Y_train.append([i])
Y_train = np.array(Y_train)
#print(Y_train)
X_train_resized = resize_data(Y_train, folder_path)
X_train = flatten_images(X_train_resized)
Y_validation = []
for i in range(6):
    img_folder = os.path.join(folder_path, f"{i}")
    for img in os.listdir(img_folder):
        Y_validation.append([i])
Y_validation = np.array(Y_validation)
X_validation_resized = resize_data(Y_validation, folder_path, True)
X_validation = flatten_images(X_validation_resized)

# Initialize the SVM classifier with Gaussian kernel
svm_classifier = SVC(kernel='rbf', gamma=0.001, C=1.0)

# Train the SVM on the training data
start_time = time.time()
svm_classifier.fit(X_train, Y_train)
training_time = time.time() - start_time

# Predict the classes for the validation set
Y_pred_b = svm_classifier.predict(X_validation)

# Calculate accuracy for part (b)
accuracy_b = np.mean(Y_pred_b == Y_validation[:,0])
print("Validation set accuracy for part (b):", accuracy_b * 100)
print("Training time for part (b):", training_time)

################ 2.2(c) #####################

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import random

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot confusion matrix for part a
plot_confusion_matrix(Y_validation, Y_pred_a, classes=np.unique(Y_validation), title='Confusion Matrix - Part (a)')

# Plot confusion matrix for part b
plot_confusion_matrix(Y_validation, Y_pred_b, classes=np.unique(Y_validation), title='Confusion Matrix - Part (b)')

# Function to visualize misclassified examples
def visualize_misclassified_examples(X_val, Y_validation, Y_pred, title):
    misclassified_indices = np.where(Y_validation != Y_pred)[0]
    selected_indices = random.sample(list(misclassified_indices), 12)

    plt.figure(figsize=(12, 10))
    plt.suptitle(title, fontsize=16)

    for i, index in enumerate(selected_indices):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X_val[index].reshape(16, 16, 3), cmap='gray', interpolation='nearest')
        plt.title(f'True: {Y_validation[index]}, Pred: {Y_pred[index]}')
        plt.axis('off')

    plt.show()

# Visualize misclassified examples for part a
visualize_misclassified_examples(X_validation, Y_validation, Y_pred_a, title='Misclassified Examples - Part (a)')

# Visualize misclassified examples for part b
visualize_misclassified_examples(X_validation, Y_validation, Y_pred_b, title='Misclassified Examples - Part (b)')

##################### 2.2 (d) #######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Load the dataset (replace with your data loading)
# For this example, let's use the Iris dataset

# Define the range of C values to test
C_values = [10**-5, 10**-3, 1, 5, 10]

# Initialize lists to store accuracies
cross_val_accuracies = []
val_accuracies = []

# Fix gamma
gamma = 0.001

# Perform 5-fold cross-validation and compute validation accuracy for each C
for C in C_values:
    # Initialize the SVM classifier with Gaussian kernel
    svm_classifier = SVC(kernel='rbf', gamma=gamma, C=C)

    # Compute 5-fold cross-validation accuracy
    cv_accuracy = np.mean(cross_val_score(svm_classifier, X_validation, Y_validation, cv=5))
    cross_val_accuracies.append(cv_accuracy)

    # Train the SVM on the training data
    svm_classifier.fit(X_train, Y_train)

    # Predict the classes for the validation set
    y_pred_val = svm_classifier.predict(X_validation)

    # Calculate validation accuracy
    val_accuracy = accuracy_score(Y_validation, y_pred_val)
    val_accuracies.append(val_accuracy)

for i in range(len(C_values)):
    print(f"The 5-fold cross validation accuracy and accuracy for C = {C_values[i]} are :", cross_val_accuracies[i] * 100, val_accuracies[i] * 100)

# Plot 5-fold cross-validation accuracy and validation set accuracy
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, cross_val_accuracies, label='5-Fold Cross-Validation Accuracy')
plt.semilogx(C_values, val_accuracies, label='Validation Set Accuracy')
plt.xlabel('C (log scale)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. C for SVM with Gaussian Kernel')
plt.grid(True)
plt.show()
