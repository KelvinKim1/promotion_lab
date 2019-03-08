from math import log

def make_naive_bayes_classfier(data):
    attribute_0 = []
    attribute_1 = []
    attributes = [attribute_0,attribute_1]
    num_rows = len(data)
    num_colm = len(data[0])

    for i in range(0,num_colm):
        attribute_0.append({})
        attribute_1.append({})

    for row in data:
        if int(row[-1]) == 0:
            for j in range(num_colm):
                if row[j] not in attribute_0[j]:
                    attribute_0[j][row[j]] = 0
                attribute_0[j][row[j]] += 1
        else:
            for j in range(num_colm):
                if row[j] not in attribute_1[j]:
                    attribute_1[j][row[j]] = 0
                attribute_1[j][row[j]] += 1

    return attributes

def naive_bayes_classify(info, attributes):
    
    class_count = [attributes[0][len(info)-1]['0'], attributes[1][len(info)-1]['1']]
    total_class = class_count[0] + class_count[1]
    prob = [log(class_count[0]/total_class),log(class_count[1]/total_class)]

    for i in range(len(info)-1):
        for j in range(2):
            if info[i] in attributes[j][i]:
                if attributes[j][i][info[i]] != 0:
                    prob[j] = log((attributes[j][i][info[i]])/class_count[j])

    if prob[0] > prob[1]:
        return 0
    return 1


