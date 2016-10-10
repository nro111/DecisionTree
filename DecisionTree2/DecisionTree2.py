import csv 
from sklearn import tree
import numpy as np


#transform data from csv into features and vectors
#this is further used to be fed into a decision tree
#that will predict survival rates on the titanic
def transformDataTitanic(trainingFile, features):
    transformData=[]
    labels = []
    
    genderMap = {"male":1,"female":2,"":""} # We include a key in the map for missing values
    embarkMap = {"C":1,"Q":2,"S":3,"":""}
    
    #Used for processing null cells in the provided csv
    blank=""
    
    with open(trainingFile,'r') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        lineNum=1
        # lineNum keeps track of the current row 
        for row in lineReader:
            if lineNum==1:
                # if it's the first row, store the names of the header in a list. 
                header = row
            else: 
                allFeatures=list(map(lambda x:genderMap[x] if row.index(x)==4
                               else embarkMap[x] if row.index(x)==11 else x, row))
                # allFeatures is a list where we have converted the categorical variables to 
                # numerical variables
                featureVector = [allFeatures[header.index(feature)] for feature in features]
                # featureVector is a subset of allFeatures, it contains only those features
                # that are specified by us in the function argument
                if blank not in featureVector:
                    transformData.append(featureVector)
                    labels.append(int(row[1]))
                    # if the featureVector contains missing values, skip it, else add the featureVector
                    # to our transformedData and the corresponding label to the list of labels
            lineNum=lineNum+1
        return transformData,labels
    # return both our list of feature vectors and the list of labels 

#by default, we use gini impurity algorithm. this can be changed by adding criterion = "entropy" as an argument to the DecisionTreeClassifier()
#this method stores the decision tree in .dot format so it can be opened by the graphviz application.
def decisionTreeClassifier(trainingData, clf):    
    x = np.array(trainingData[0])
    y = np.array(trainingData[1])
    clf = clf.fit(x, y)
    with open("titanic.dot", "w") as function:
        function = tree.export_graphviz(clf,
                                        feature_names = features,
                                        class_names = ["Dead", "Survived"],
                                        filled = True, rounded = True, special_characters = True, out_file = function)

def transformTestDataTitanicv2(testFile,features):   
    transformData=[]
    ids=[]   
    genderMap={"male":1,"female":2,"":1} # Map blanks to males
    embarkMap={"C":1,"Q":2,"S":3,"":3} # Map the default port of embarkation to Southampton
    blank=""
    with open(testFile,"r") as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        lineNum=1
        for row in lineReader:
            if lineNum==1:
                header=row
            else: 
                allFeatures=list(map(lambda x:genderMap[x] if row.index(x)==3 else embarkMap[x] 
                               if row.index(x)==10 else x,row))              
                # The second column is Passenger class, let the default value be 2nd class
                if allFeatures[1]=="":
                    allFeatures[1]=2
                # Let the default age be 30
                if allFeatures[4]=="":
                    allFeatures[4]=30
                # Let the default number of companions be 0 (assume if we have no info, the passenger
                # was travelling alone)
                if allFeatures[5]=="":
                    allFeatures[5]=0
                # By eyeballing the data , the average fare seems to be around 30
                if allFeatures[8]=="":
                    allFeatures[8]=32
                featureVector=[allFeatures[header.index(feature)] for feature in features]     
                transformData.append(featureVector)
                ids.append(row[0])
            lineNum=lineNum+1 
    return transformData,ids

def titanicTest(classifier, resultFile, transformDataFunction = transformTestDataTitanicv2):
    testData = transformDataFunction(testFile, features)
    result = classifier.predict(testData[0])
    with open(resultFile,"w") as f:
        ids=testData[1]
        lineWriter=csv.writer(f,delimiter=',',quotechar="\"")
        lineWriter.writerow(["PassengerId","Survived"])#The submission file needs to have a header
        for rowNum in range(len(ids)):
            try:
                lineWriter.writerow([ids[rowNum],result[rowNum]])
            except(Exception,e):
                print(e)

clf = tree.DecisionTreeClassifier(min_samples_split = 85, max_leaf_nodes=8, min_impurity_split=0.2, splitter="best", presort=True)
trainingFile="C:/Users/normsby/Documents/datasets/train.csv"
testFile="C:/Users/normsby/Documents/datasets/test.csv"
features=["Pclass","Sex","Age","SibSp","Fare"]
trainingData=transformDataTitanic(trainingFile,features)
decisionTreeClassifier(trainingData, clf)
resultFile="C:/Users/normsby/Documents/datasets/result3.csv"
titanicTest(clf,resultFile,transformTestDataTitanicv2)