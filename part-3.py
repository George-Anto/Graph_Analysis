import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix, 
                             accuracy_score, top_k_accuracy_score)
from sklearn.ensemble import RandomForestClassifier 

def read_data(path = "data/10000_rows_N10", train_or_test_data = "1st_subgraph"):
    features_df = pandas.read_csv(path + train_or_test_data + "_data.csv", sep = " ", index_col=False)
    #If there are columns that specify the source and target nodes, 
    #we erase them so they do not interfere with the classification training algorithm 
    features_df.drop(['source', 'target'], axis = 1, inplace = True, errors='ignore')
    #Attempt to normalize the metrics from the previous question,
    #our test data gat worse metrics (mse, r2, score) if we make this operation    
    #features_df=(features_df-features_df.mean())/features_df.std()
    labels_df = pandas.read_csv(path + train_or_test_data + "_existing_edges.csv", sep = " ", index_col=False)
    #If there are columns that specify the source and target nodes, 
    #we erase them so they do not interfere with the classification training algorithm 
    labels_df.drop(['source', 'target'], axis = 1, inplace = True, errors='ignore')
    
    return [features_df, labels_df]

def calc_metrics(cm):
    #Get the following information from the train confusion matrix
    true_positive = cm[0][0]
    false_negative = cm[0][1]
    false_positive = cm[1][0]
    true_negative = cm[1][1]
    #Calculate the following 3 metrcis for our data
    #Precision is the measure of true positives over the number of total 
    #positives predicted by our model. What this metric allows us to calculate
    #is the rate of which our positive predictions are actually positive.
    precision = true_positive / (true_positive + false_positive)
    #Recall (a.k.a sensitivity) is the measure of our true positive over the
    #count of actual positive outcomes. Using this metric, we can assess how 
    #well our model is able to identify the actual true result.
    recall = true_positive / (true_positive + false_negative)
    #The F1 score is the harmonic mean between precision and recall.
    #This score can be used as an overall metric that incorporates both 
    #precision and recall. The reason we use the harmonic mean as opposed 
    #to the regular mean, is that the harmonic mean punishes values that 
    #are further apart.
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return [precision, recall, f1_score]

#Manipulate the path to be able to read data from the desired location
path = "data/10000_rows_N10/"

#We use the matrics and the existing edges from the 1st subgraph for train data
train_data = read_data(path, "1st_subgraph")
features_train_df = train_data[0]
labels_train_df = train_data[1]

#We use the matrics and the existing edges from the 2nd subgraph for test data
test_data = read_data(path, "2nd_subgraph")
features_test_df = test_data[0]
labels_test_df = test_data[1]

# Get from the user debug or production mode
# #Create a loop to ensure tha the user will give a valid input 1 or 0
while True:
    mode =input("\n 1 = Production MODE, not much printing \n 0 = Debug MODE with more printing \n Give the MODE : ")
    try:
        #If the input is not 1 or 0, the code in the except block will be executed
        #and the user will be prompted to enter a new value 
        mode = int(mode)
        #If the number is not 1 or 0, the user will have to enter a new value too
        if mode != 0 and mode != 1:  # if not 1 or 0 ask for input again
            print("\nSorry, input must be 1 or 0, try again.")
            continue
        break
    except ValueError:
        print("\nThat's not a correct value, try again.") 

#If the user selected debug mode, we print the following information to them
if mode == 0:
    print("\n---------- Train Dataframes ----------\n")
    print("Features Dataframe:\n", features_train_df)
    print(features_train_df.shape)
    print("\nLabels Dataframe:\n", labels_train_df)
    print(labels_train_df.shape)
    print("\n---------- Test Dataframes ----------\n")
    print("Features Dataframe:\n", features_test_df)
    print(features_test_df.shape)
    print("\nLabels Dataframe:\n", labels_test_df)
    print(labels_test_df.shape)
    print("\n--------------------------------------\n")

#The classifier we use to train the data and classifie the labels to 1 or 0
#1 means that the edge between 2 nodes exists and 0 that the edge is absent
clf = RandomForestClassifier(max_depth = 10)
#Train the network using the training data (1st subgraph data)
clf.fit(features_train_df, labels_train_df.values.ravel())

#The importance metric shows the weight each of our 5 metrics have in the final training
importance_metrics = clf.feature_importances_ 
print("\n---------- Metrics Importance ----------\n")
print("Graph Distance: ", importance_metrics[0])
print("Common Neighbors: ", importance_metrics[1])
print("Jaccard Coefficient: ", importance_metrics[2])
print("Adamic - Adar: ", importance_metrics[3])
print("Preferential Attachment: ", importance_metrics[4])
print("\n----------------------------------------\n")

#Calculate the predictions about the class of the labels our algorithm produces
#We calculate the predictions both for our train and our test data
predicted_train_labels = clf.predict(features_train_df)
predicted_test_labels = clf.predict(features_test_df)

#Calculate the mean squared error and the r2 score dor our train data
train_mse = mean_squared_error(labels_train_df, predicted_train_labels)
train_r2 = r2_score(labels_train_df, predicted_train_labels)
#Calculate also the Model Accuracy, how often is the classifier correct for the train data
#we calculate the accuracy both in a percent and in absolute numbers 
train_accuracy_percent = accuracy_score(labels_train_df, predicted_train_labels)
train_accuracy_quantity = top_k_accuracy_score(labels_train_df, predicted_train_labels, k = 1, normalize=False)

#Calculate the Confusion Matrix for the train data
cm_train = confusion_matrix(predicted_train_labels, labels_train_df)
#Calculate precision, recall and f1 score for the train data
train_metrics = calc_metrics(cm_train)

#Calculate the mean squared error and the r2 score dor our test data                  
test_mse = mean_squared_error(labels_test_df, predicted_test_labels)
test_r2 = r2_score(labels_test_df, predicted_test_labels)
#Calculate also the Model Accuracy, how often is the classifier correct for the test data
#we calculate the accuracy both in a percent and in absolute numbers
test_accuracy_percent = accuracy_score(labels_test_df, predicted_test_labels)
test_accuracy_quantity = top_k_accuracy_score(labels_test_df, predicted_test_labels, k = 1, normalize=False)

#Calculate the Confusion Matrix for the test data
cm_test = confusion_matrix(predicted_test_labels, labels_test_df)
#Calculate the 3 metrics for our test data also
test_metrics = calc_metrics(cm_test)

#Print all the different metrics to the console
#for both the train and the test data 
print('MSE (Train): ', train_mse)
print('R2 (Train): ', train_r2)
print("Accuracy (%) (Train):", train_accuracy_percent)
print("Accuracy (Quantity) (Train):", train_accuracy_quantity)
print("\nConfusion Matrix (Train)\n", cm_train)
print("\nPrecision (Train): ", train_metrics[0])
print("Recall (Train): ", train_metrics[1])
print("F1 Score (Train): ", train_metrics[2])
print('- - - - - - - - - - - - - -')
print('MSE (Test): ', test_mse)
print('R2 (Test): ', test_r2)
print("Accuracy (%) (Test):", test_accuracy_percent)
print("Accuracy (Quantity) (Test):", test_accuracy_quantity)
print("\nConfusion Matrix (Test)\n", cm_test)
print("\nPrecision (Test): ", test_metrics[0])
print("Recall (Test): ", test_metrics[1])
print("F1 Score (Test): ", test_metrics[2])

#Calculate the accurace of our predictions for both the train and the test data
#Using another and final random forest classifier method
train_accurace_score = clf.score(features_train_df, labels_train_df)
test_accurace_score = clf.score(features_test_df, labels_test_df)

#Print them also
print("\n---------- Final Accurace Score ----------")
print("\nTrain Accurace Score: ", train_accurace_score, "/ 1.0")
print("Test Accurace Score: ", test_accurace_score, "/ 1.0\n")

#Styling for the plots
train_font = {'family':'serif','color':'#7CAE00','size':15}
test_font = {'family':'serif','color':'#ae3200','size':15} 

#Create a plot that shows the actual class for the data in the x axis
#and the predicted class in the y axis
#Train data feeeded
plt.scatter(x = labels_train_df, y = predicted_train_labels, c="#7CAE00" ,alpha=0.3)
plt.title('Train Data - Random Forest Classification', fontdict=train_font)
plt.ylabel('Predicted Class', fontdict=train_font)
plt.xlabel('Actual Class', fontdict=train_font)
plt.show()

#Create a plot that shows the actual class for the data in the x axis
#and the predicted class in the y axis
#Test data feeded
plt.scatter(x = labels_test_df, y = predicted_test_labels, c="#ae3200" ,alpha=0.3)
plt.title('Test Data - Random Forest Classification', fontdict=test_font)
plt.ylabel('Predicted Class', fontdict=test_font)
plt.xlabel('Actual Class', fontdict=test_font)
plt.show()