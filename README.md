# APS1070 Project 4 - Linear Regression, Batch and Gradient Descent
-	Use of Panda: pd.DataFrame; pd.read_csv; pd.merge
-	Use of Numpy: np.dot; np.random.permutation (shuffle)
-	Use of matplotlib.pyplot: plt.plot;
-	Use of sklearn:
sklearn.model_selection - mean_squared_error
sklearn.preprocessing – StandardScaler - inverse_transform
sklearn.metrics - roc_auc_score - f1_score - precision_score - recall_score
-	Use of scipy: linalg.inv
-	Python Basics: .iloc; .loc; .insert

Part 1 – Data Preparation
-	Goal is to design a mathematical model that predicts electrical grid stability
-	Standardize and insert column 1 for the data

Part 2 – Linear Regression Model
-	Beta = np.dot(inv(np.dot(X_train_std.T, X_train_std)), np.dot(X_train_std.T, y_train))

Part 3 – Full Batch Gradient Descent
-	The whole dataset is 1 batch
-	One epoch is one iteration
-	Define a function to record the operation time, training and validation RMSE

Part 4 - Mini-batch and Stochastic Gradient Descent
-	Mini batch breaks the dataset to k smaller batches; One epoch takes k iteration.
-	Stochastic batch make each sample as a batch; One epoch takes n iteration.
-	Modify the function of part3 to make it a mini-batch case.
-	Key point: after each epoch, you need to shuffle the entire training set.
-	Trade-off between the speed of convergence with small batches and the stability of convergence with large batches. 
-	Batch size will significantly affect the convergence speed with respect to time. While larger batches will take a shorter time because they reduce the number of required updates.

Part 5 - Learning Rate 
-	Select the best batch size based on Part 4's fastest convergence time and sweep the learning rate
-	Using a larger learning rate reduces the time and number of epochs needed for convergence.
