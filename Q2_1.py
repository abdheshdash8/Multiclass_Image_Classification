import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import cv2
import os

class SVM_binary_classifier:
    def __init__(self, folder_path, class_1, class_2, C = 1):
        self.path = folder_path
        self.class_1 = class_1
        self.class_2 = class_2
        self.C = C

    def flatten_images(self, data):
        data = data / 255
        flattened_images = data.reshape(data.shape[0], -1)
        return flattened_images
    
    def resize_data(self, class_label):
        images = []
        img_folder = os.path.join(self.path, f"{class_label}(1)")
        for img in os.listdir(img_folder):
            path = os.path.join(img_folder, img)
            img_data = cv2.imread(path)
            #rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img_data, (16, 16))
            images.append(resized_image)
        return np.array(images)
    
    def quad_optimize(self, X, Y):
        # m is the no. of training samples.
        self.m = X.shape[0]
        # n is the dimension of X i.e. 768.
        self.n = X.shape[1]
        P = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                P[i, j] = Y[i] * Y[j] * np.dot(X[i], X[j])

        q = matrix(-np.ones(self.m), tc = "d")
        G = matrix(np.vstack((-np.eye(self.m), np.eye(self.m))), tc = "d")
        h = matrix(np.hstack((np.zeros(self.m), self.C * np.ones(self.m))), tc = "d")
        A = matrix(Y.reshape(1, -1), tc = "d")
        b = matrix(np.zeros(1), tc = "d")
        sol = solvers.qp(matrix(P, tc = "d"), q, G, h, A, b)["x"]
        alphas = np.array(sol).reshape(self.m)

        # Compute the support vectors
        support_vectors = X[alphas > 1e-6]
        support_vector_alphas = alphas[alphas > 1e-6]
        support_vector_labels = Y[alphas > 1e-6]
        num_supp_vecs = len(support_vectors)
        print("Percentage of training data that are support vectors", (num_supp_vecs / self.m) * 100)
        return alphas, support_vectors, support_vector_alphas, support_vector_labels
    
    def calc_w_b(self, X, Y, alphas):
        # Calculate weight vector w
        w = ((Y * alphas).T @ X).reshape(-1, 1)
        # Calculate intercept b
        b = 0
        for supp_vec_ind in np.where(alphas > 1e-6)[0]:
            b += Y[supp_vec_ind] - sum(alphas * Y * np.dot(X, X[supp_vec_ind].T))
        b /= len(np.where(alphas > 1e-6)[0])

        return w, b
     
    def load_validation_data(self):
        validation_images = []
        Y_validation = np.array([])
        for i in range(2):  # For taking validation of the 2 classes 5 and 0
            img_folder = os.path.join(self.path, f"{(i-1)%6}")
            for img in os.listdir(img_folder):
                path = os.path.join(img_folder, img)
                img_data = cv2.imread(path)
                #rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(img_data, (16, 16))
                validation_images.append(resized_image)
                if i == 0:
                    Y_validation = np.append(Y_validation, i+1)
                else:
                    Y_validation = np.append(Y_validation, -i)
        return np.array(validation_images), Y_validation
    
    def plot_supp_vecs_1(self, support_vectors, support_vector_alphas, w):
        six_max_alphas = np.argpartition(support_vector_alphas, -6)[-6:]
        # Reshape support vectors and weight vector for plotting
        reshaped_support_vectors = support_vectors[six_max_alphas].reshape(-1, 16, 16, 3)
        reshaped_weight_vectors = w.reshape(16, 16, 3)
        
        plt.imshow(reshaped_weight_vectors)
        plt.title('Weight Vector')
        plt.show()

        # For plotting the reshaped support vectors
        for i in range(6):
            plt.imshow(reshaped_support_vectors[i])
            plt.title(f'Support Vector {i+1}')
            plt.show()

    def gaussian_kernel_optimize(self, X, Y, gamma):
        self.m = X.shape[0]
        self.n = X.shape[1] 
        P = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                P[i, j] = Y[i] * Y[j] * np.exp((-gamma) * np.linalg.norm(X[i]-X[j])**2)
        q = matrix(-np.ones(self.m))
        G = matrix(np.vstack((-np.eye(self.m), np.eye(self.m))))
        h = matrix(np.hstack((np.zeros(self.m), self.C * np.ones(self.m))))
        A = matrix(Y.reshape(1, -1))
        b = matrix(np.zeros(1))
        sol = solvers.qp(matrix(P), q, G, h, A, b)
        alphas = np.array(sol['x']).reshape(self.m)
        # Compute the support vectors
        support_vectors = X[alphas > 1e-6]
        support_vector_alphas = alphas[alphas > 1e-6]
        support_vector_labels = Y[alphas > 1e-6]
        num_supp_vecs = len(support_vectors)
        print("Percentage of training data that are support vectors", (num_supp_vecs / self.m) * 100)
        return alphas, support_vectors, support_vector_alphas, support_vector_labels, num_supp_vecs
    
    def plot_supp_vecs_gaussian(self, support_vectors, support_vector_alphas):
        six_max_alphas = np.argpartition(support_vector_alphas, -6)[-6:]
        reshaped_support_vectors = support_vectors[six_max_alphas].reshape(-1, 16, 16, 3)
        # For plotting the reshaped support vectors
        for i in range(6):
            plt.imshow(reshaped_support_vectors[i])
            plt.title(f'Support Vector {i+1}')
            plt.show()

    def classify_using_kernel(self, X_validation, alphas, X, Y, gamma, support_vectors, support_vector_labels, num_supp_vecs):
        Y_pred = np.zeros(X_validation.shape[0])
        supp_vec_ind = np.where(alphas > 1e-4)[0]
        vec = np.zeros(num_supp_vecs)
        for supp_vec_ind in np.where(alphas > 1e-4)[0]:
            vec = vec + Y[supp_vec_ind]*alphas[supp_vec_ind]*np.exp(-gamma* np.sum((support_vectors - X[supp_vec_ind])*(support_vectors - X[supp_vec_ind]), axis = 1))
        b = np.mean(support_vector_labels - vec)
        for i in range(X_validation.shape[0]):
            prediction = b
            for j in range(X.shape[0]):
                prediction += alphas[j] * Y[j] * np.exp((-gamma) * np.linalg.norm(X[j] - X_validation[i])**2)
            Y_pred[i] = np.sign(prediction)
            
        return Y_pred

class_1 = 5     ## As my entry no. end with 5
class_2 = 0
folder_path = 'C:\\Users\\Acer\\Desktop\\IITD\\3rd year\\Sem. 1\\COL774\\a2\\Q2'
binary_classifier = SVM_binary_classifier(folder_path, class_1, class_2)
X_train_1_resized = binary_classifier.resize_data(class_1)
X_train_1 = binary_classifier.flatten_images(X_train_1_resized)
X_train_2_resized = binary_classifier.resize_data(class_2)
X_train_2 = binary_classifier.flatten_images(X_train_2_resized)
X = np.vstack((X_train_1, X_train_2))
#print(X)
Y_train_1 = np.ones(X_train_1.shape[0])
Y_train_2 = -np.ones(X_train_2.shape[0])
Y = np.hstack((Y_train_1, Y_train_2))
#print(Y)

### Loading the validation dataset ###
validation_images, Y_validation = binary_classifier.load_validation_data()
validation_data = binary_classifier.flatten_images(validation_images)
#print(Y_validation)

l1 = binary_classifier.quad_optimize(X, Y)
w, b = binary_classifier.calc_w_b(X, Y, l1[0])
#print(w)

y_pred = np.sign(np.dot(validation_data, w) + b)
#print(y_pred)

# Calculating validation accuracy
validation_accuracy = 0
for i in range(400):
    if y_pred[i][0] == Y_validation[i]:
        validation_accuracy += 1
validation_accuracy /= 400
print(f"Validation set accuracy: {validation_accuracy * 100:.4f}")

binary_classifier.plot_supp_vecs_1(l1[1], l1[2], w)

################### Q2.1 (b) ######################

gamma = 0.001
binary_classifier = SVM_binary_classifier(folder_path, class_1, class_2)
l1 = binary_classifier.gaussian_kernel_optimize(X, Y, gamma)


# Assuming y_validation contains the true labels for the validation set
Y_pred = binary_classifier.classify_using_kernel(validation_data, l1[0], X, Y, gamma, l1[1], l1[3], l1[4])
#print(Y_pred)
# Calculate validation accuracy
validation_accuracy = 0
for i in range(400):
    if Y_pred[i] == Y_validation[i]:
        validation_accuracy += 1
validation_accuracy /= 400

print("Validation set accuracy:", validation_accuracy * 100)

binary_classifier.plot_supp_vecs_gaussian(l1[1], l1[2])

################## Q2.1 (c) #######################

from sklearn import svm
import time

#### Part(i) ####

# Linear Kernel 
start_time_1 = time.time()
linear_training_time = time.time() - start_time_1
linear_svm = svm.SVC(kernel = "linear", C = 1.0)
Y = []
for i in range(2380):
    Y.append(1)
for i in range(2380):
    Y.append(-1)
Y = np.array(Y)
Y_validation = []
for i in range(200):
    Y_validation.append(1)
for i in range(200):
    Y_validation.append(-1)
Y_validation = np.array(Y_validation)

linear_svm.fit(X, Y)
nSV_1 = np.sum(linear_svm.n_support_)
w = linear_svm.coef_.flatten()
b = linear_svm.intercept_[0]

linear_training_time = time.time() - start_time_1

# Gaussian Kernel
start_time_2 = time.time()
gaussian_svm = svm.SVC(kernel='rbf', C = 0.1, gamma= 0.001)
gaussian_svm.fit(X, Y)
nSV_2 = np.sum(gaussian_svm.n_support_)
gaussian_training_time = time.time() - start_time_2

print("Number of Support Vectors for Linear Kernel (nSV1):", nSV_1)
print("Number of Support Vectors for Gaussian Kernel (nSV2):", nSV_2)

#### Part-(ii) ####

y_pred_linear = linear_svm.predict(validation_data)
linear_accuracy = np.mean(y_pred_linear == Y_validation)

y_pred_gaussian = gaussian_svm.predict(validation_data)
gaussian_accuracy = np.mean(y_pred_gaussian == Y_validation)

#### Part-(iii) ####
print("Validation Accuracy for Linear Kernel:", linear_accuracy*100)
print("Validation Accuracy for Gaussian Kernel:", gaussian_accuracy*100)

#### Part-(iv) ####

print("Training Time Linear Kernel with scikit-learn:", linear_training_time)
print("Training Time Gaussian Kernel with scikit-learn:", gaussian_training_time)