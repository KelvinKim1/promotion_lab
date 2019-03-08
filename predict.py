import regression_tree
import Bayes_classifier
import sys
import csv

data = []

with open("training_set.csv") as csvfile, open("test_set.csv") as csvfile2, open("predicted.csv","w") as t:
    readCSV = csv.reader(csvfile, delimiter=',')
    readCSV2 = csv.reader(csvfile2, delimiter=',')
    temp = csv.writer(t,delimiter=',')
    for row in readCSV:
        data.append(list(row))
        
    tree = regression_tree.buildtree(data, min_gain = 0.001, min_samples = 5)
    result = Bayes_classifier.make_naive_bayes_classfier(data)

    instance = 0
    count1 = 0
    count2 = 0
    for row in readCSV2:
        instance+=1
        probability = regression_tree.classify(row,tree)
        if probability < 0.5:
            predicted1 = 0
        else:
            predicted1 = 1
            
        predicted2 = Bayes_classifier.naive_bayes_classify(row,result)

        if predicted1 == int(row[-1]):
            count1+=1
        if predicted2 == int(row[-1]):
            count2+=1

        output = [instance,row[-1],predicted1, probability]
        temp.writerow(output)
    print("Accuracy of decision tree is ")
    print(count1/instance)
    print("Accuracy of naive Bayes classfier is ")
    print(count2/instance)


csvfile.close()
csvfile2.close()
t.close()




