import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def read_data_reg_file():
     #open the file
    file = open('data_reg.csv')
    # read the file
    data = pd.read_csv(file)
    return data

def plot_training_validation_testing_data(data):
     # training, validation, testing
    training_set = []
    validation_set = []
    testing_set = []
    
    # split data into training, validation, testing
    training_set = data.iloc[0:120]
    validation_set = data.iloc[120:160]
    testing_set = data.iloc[160:]
  

    # plot the three data set, each with different color
    # create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')

    # plot training set
    ax.scatter(training_set['x1'], training_set['x2'], training_set['y'],c='r',marker='o',label = 'Training set')
    #plot validation set
    ax.scatter(validation_set['x1'], validation_set['x2'], validation_set['y'],c='b',marker='^',label = 'Validation set')
    # plot testing set
    ax.scatter(testing_set['x1'], testing_set['x2'], testing_set['y'],c='g',marker='s',label = 'Testing set')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.legend()
    plt.show()

def polynomial_regression(data,train_set,validation_set,test_set):
   
    # Extract features and target from the training set
    X_train = train_set[['x1', 'x2']]
    y_train = train_set['y']

    # Extract features and target from the validation set
    X_val = validation_set[['x1', 'x2']]
    y_val = validation_set['y']

    X_test = test_set[['x1','x2']]
    Y_test = test_set['y']


    # Initialize lists to store validation errors for each degree
    degrees = list(range(1, 11))
    validation_errors = []


    # Loop through polynomial degrees and fit models
    for degree in degrees:
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_val_poly = poly_features.transform(X_val)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Make predictions on the validation set
        y_val_pred = model.predict(X_val_poly)

        # Calculate validation error (MSE)
        mse = mean_squared_error(y_val, y_val_pred)
        validation_errors.append(mse)
        #Plot_surface_of_the_learned_function(train_set,X_train,poly_features,model,degree)

       

    # Plot validation error vs polynomial degree curve
    plt.plot(degrees, validation_errors, marker='o')
    plt.title('Validation Error vs Polynomial Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Validation Error (MSE)')
    plt.show()
    
    # Find the best polynomial degree (minimum validation error)
    best_degree = degrees[np.argmin(validation_errors)]

    # test the best poly
    polynomial_features = PolynomialFeatures(degree=best_degree)
    X_train_Poly = polynomial_features.fit_transform(X_train)
    X_test_poly = polynomial_features.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_train_Poly,y_train)

    Y_test_pred = model.predict(X_test_poly)

    mse = mean_squared_error(Y_test, Y_test_pred)
    print(mse)


   


    
def Plot_surface_of_the_learned_function(train_set,X_train,poly_features,model,degree):
     # Plot surface of the learned function alongside training examples
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for training set
        ax.scatter(train_set['x1'], train_set['x2'], train_set['y'], c='blue', marker='o', label='Training Set')

        # Plot surface of the learned function
        x1_range = np.linspace(min(X_train['x1']), max(X_train['x1']), 100)
        x2_range = np.linspace(min(X_train['x2']), max(X_train['x2']), 100)
        x1_vals, x2_vals = np.meshgrid(x1_range, x2_range)
        X_surface = np.c_[x1_vals.ravel(), x2_vals.ravel()]
        X_surface_poly = poly_features.transform(X_surface)
        y_surface = model.predict(X_surface_poly)
        y_surface = y_surface.reshape(x1_vals.shape)
        ax.plot_surface(x1_vals, x2_vals, y_surface, alpha=0.5, cmap='viridis', label=f'Degree {degree} Fit')

        # Set axis labels
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.title(f'Polynomial Degree {degree}')
        plt.show()

def ridge_regression(X_train,y_train,X_val):
      # Polynomial degree
    degree = 8

    # Regularization parameters
    alphas = [0.001, 0.005, 0.01, 0.1, 10]

    # List to store mean squared errors for each alpha
    mse_values = []

    # Apply polynomial regression with Ridge regularization for each alpha
    for alpha in alphas:
        # Apply polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        
        # Apply Ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_poly, y_train)
        
        # Make predictions on validation set
        y_val_pred = ridge.predict(X_val_poly)
        
        # Calculate mean squared error
        mse = mean_squared_error(y_val, y_val_pred)
        mse_values.append(mse)
    plot_ridge_regression_parameter(alphas,mse_values,X_train_poly)

def plot_ridge_regression_parameter(alphas,mse_values,X_train_poly):
    # Plot MSE on validation vs regularization parameter
    plt.plot(alphas, mse_values, marker='o')
    plt.xscale('log')  # Use log scale for better visualization
    plt.title('MSE on Validation vs Regularization Parameter')
    plt.xlabel('Regularization Parameter (alpha)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.show()

    # Find the best alpha (minimize MSE)
    best_alpha = alphas[np.argmin(mse_values)]
    print(f"Best Regularization Parameter (alpha): {best_alpha}")

    # Apply Ridge regression with the best alpha on the entire training set
    ridge_best = Ridge(alpha=best_alpha)
    ridge_best.fit(X_train_poly, y_train)
     
def Logistic_Regression():

    # read train file
    train_data = pd.read_csv("train_cls.csv")
    test_data = pd.read_csv("test_cls.csv")

    # the target is string, we need to convert it to binary
    Label_Encoder = LabelEncoder()
    train_data['class'] = Label_Encoder.fit_transform(train_data['class'])
    test_data['class'] = Label_Encoder.fit_transform(test_data['class'])
    # C1 => ZERO
    # C2 => ONE

    model = LogisticRegression()
    # model learning
    model.fit(train_data[['x1','x2']], train_data['class'])
    # model testing
    y_pred = model.predict(test_data[['x1','x2']])
    accuracy = accuracy_score(test_data['class'],y_pred)
    print(f'accuracy = {accuracy}')

    h = .02  # step size in the mesh
    X_train = train_data[['x1','x2']]
    x_min, x_max = X_train['x1'].min() - 1, X_train['x1'].max() + 1
    y_min, y_max = X_train['x2'].min() - 1, X_train['x2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_train['x1'], X_train['x2'], c=train_data['class'], cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def  Logistic_Regression_part2():
    # read train file
    train_data = pd.read_csv("train_cls.csv")
    test_data = pd.read_csv("test_cls.csv")

      # the target is string, we need to convert it to binary
    Label_Encoder = LabelEncoder()
    train_data['class'] = Label_Encoder.fit_transform(train_data['class'])
    test_data['class'] = Label_Encoder.fit_transform(test_data['class'])
    # C1 => ZERO
    # C2 => ONE



    poly_feature = PolynomialFeatures(degree=2)
    X_train = poly_feature.fit_transform(train_data[['x1','x2']])
    X_test = poly_feature.fit_transform(test_data[['x1','x2']])
    Y_train = train_data['class']
    Y_test = test_data['class']

    model = LogisticRegression()
    # model learning
    model.fit(X_train, Y_train)
    # model testing
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred)
    print(f'accuracy = {accuracy}')
    plot_decision_boundary(X_train, Y_train, model, poly_feature)
    
    

def plot_decision_boundary(X, Y, model, poly_feature):
    h = .02  # step size in the mesh

    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(poly_feature.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 1], X[:, 2], c=Y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()    




# check if the file exists
if os.path.exists('data_reg.csv'):

    # open the first file and read data from it
    data = read_data_reg_file()

    # plot the data
    #plot_training_validation_testing_data(data)

   
    # Split the data into training and validation sets
    train_set = data[:120]
    validation_set = data[120:160]
    test_set = data[160:]
    # Extract features and target from the training set
    X_train = train_set[['x1', 'x2']]
    y_train = train_set['y']
    # Extract features and target from the validation set
    X_val = validation_set[['x1', 'x2']]
    y_val = validation_set['y']

     # plynomial regression
    #polynomial_regression(data,train_set,validation_set,test_set)
    #ridge_regression(X_train,y_train,X_val)
    #######################
    # part two of the assingment
    #Logistic_Regression()
    Logistic_Regression_part2()



        
        
else:
    print("FILE DOES NOT EXIST")    