import math
import sys

def readfile(inputfile):
    d = list()
    data = dict()
    with open(inputfile,'r') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            value = line.strip().split('\t')
            d.append(value)
    for elem1 in range(len(header)):
        for elem2 in range(len(d)):
            if not header[elem1] in data.keys():
                data[header[elem1]] = [d[elem2][elem1]]
            else:
                data[header[elem1]].append(d[elem2][elem1])
    return data

def attributes(data):
    attributes = dict()
    for item in data:
        for i in data[item]:
            if not item in attributes:
                attributes.update({item:[i]})
            elif not i in attributes[item]:
                attributes[item].append(i)
    return attributes

max_depth = 5 #sys.argv[]
Label1 = list(attributes(readfile('education_train.tsv')).values())[0][0]
label2 = list(attributes(readfile('education_train.tsv')).values())[0][1]
label3 = list(attributes(readfile('education_train.tsv')).values())[-1][0]
label4 = list(attributes(readfile('education_train.tsv')).values())[-1][1]

def counts(data,attributes):
    counts = dict()
    for item in data:
        x = list()
        for i in attributes[item]:
            x.append(data[item].count(i))
        if not item in counts:
            counts.update({item:x})     
    return counts                

def gini_y(counts):
    for item in counts:
        if item == list(counts.keys())[-1]:
            impurity = 1
            for i in counts[item]:
                val = (i/sum(counts[item]))**2
                impurity = impurity - val
    return impurity

def gini_imp(probs):
    gini = dict()
    for item in probs:
        impurity = 1
        for i in probs[item]:
            impurity = impurity - (i)**2
        if not item in gini:
            gini.update({item:impurity})
    return gini

def probability(counts):
    prob = dict()
    for item in counts:
        for i in range(len(counts[item])):
            if not item in prob:
                prob.update({item:[counts[item][i]/sum(counts[item])]})
            else:
                prob[item].append(counts[item][i]/sum(counts[item]))
    for item in prob:
        if prob[item]==[1.0]:
            prob[item] = [1.0,0.0]
    return prob

def gini_gain(data,attributes_,gini_imp_y_,probability_):
    d1 = dict()
    d2 = dict()
    for item in data:
        if item in list(data.keys())[:-1]:
            for i in range(len(data[item])):
                if data[item][i] == attributes_[item][0]:
                    if not item in d1:
                        d1.update({item:[data[list(data.keys())[-1]][i]]})
                    else:
                        d1[item].append(data[list(data.keys())[-1]][i])

                elif data[item][i] == attributes_[item][1]:
                    if not item in d2:
                        d2.update({item:[data[list(data.keys())[-1]][i]]})
                    else:
                        d2[item].append(data[list(data.keys())[-1]][i])
    
    for item in d1:
        if not item in d1:
            d1[item] = 0
        
##     Split based on column features to calculate gini impurities
##     print(d1,d2)
    local_attributes1 = attributes(d1)
##     print(local_attributes1)
    local_counts1 = counts(d1,local_attributes1)
##     print(local_counts1)
    local_probability1 = probability(local_counts1)
##     print(local_probability1)
    local_gini1 = gini_imp(local_probability1)
##     print(local_gini1)
    local_attributes2 = attributes(d2)
##     print(local_attributes2)
    local_counts2 = counts(d2,local_attributes2)
##     print(local_counts2)
    local_probability2 = probability(local_counts2)
##     print(local_probability2)
    local_gini2 = gini_imp(local_probability2)
##     print(local_gini2)
    
##     To combine the two local_variables      
    for item in d2:
        if not item in d2:
            d2[item] = 0
    
    local_gini = {**local_gini1, **local_gini2}
    for key, value in local_gini.items():
        if key in local_gini1 and key in local_gini2:
            local_gini[key] = [local_gini1[key],value]
    
    for key in local_gini:
        if local_gini[key] == 0.0:
            local_gini[key] = [0.0,0.0]
        elif type(local_gini[key])==float:
            local_gini[key] = [local_gini[key],0.0]
            
##     To calculate products in Gini Gain function
    prod = dict()
    for item in local_gini:
        for i in range(len(local_gini[item])):
            if item not in prod:
                prod.update({item:[probability_[item][i]*local_gini[item][i]]})
            else:
                prod[item].append(probability_[item][i]*local_gini[item][i])

##     To calculate the final gini gain for each column
    g_gain = dict()
    for item in prod:
        gain = gini_imp_y_
        for j in range(len(prod[item])): 
            gain = gain-prod[item][j]
        if not item in g_gain:
            g_gain.update({item:gain})
    return g_gain

def max_gain(gini_gain_):
    flip = [k for k,v in gini_gain_.items() if v==max(gini_gain_.values())]
    if len(flip)>0:
        flip_val = gini_gain_[flip[0]]
        flip.insert(0,flip_val)
    return flip

def split(data,flip_,attributes_):
    d1 = dict()
    d2 = dict()
    for i in range(len(data[flip_])):
        if data[flip_][i] == attributes_[flip_][0]:
            for item in data:
                if not item in d1:
                    d1.update({item:[data[item][i]]})
                else:
                    d1[item].append(data[item][i])
        elif data[flip_][i] == attributes_[flip_][1]:
            for item in data:
                if not item in d2:
                    d2.update({item:[data[item][i]]})
                else:
                    d2[item].append(data[item][i])
    
    if flip_ in d1:
        del d1[flip_]
    if flip_ in d2:
        del d2[flip_]
    
    return [d1,d2]

def repeat(t):
    s = attributes(t)
    r = counts(t,s)
    q = gini_y(r)
    p = probability(r)
    o = gini_gain(t,s,q,p)
    n = max_gain(o)
    m = split(t,n[1],s)
    
    return m+n

def maj_vote(data,attribute):
    temp = attributes(data)
    temp2 = split(data,attribute,temp)
    temp3 = majority_vote(temp2[0])
    temp4 = majority_vote(temp2[1])
    return[temp3,temp4]

def majority_vote(data):
    c1 = data[list(data.keys())[-1]].count('democrat')
    c2 = data[list(data.keys())[-1]].count('republican')
    
    if c1>c2:
        return 'democrat'
    else:
        return 'republican'

class TreeNode:
    def __init__(self):
        self.name = None
        self.val=None
        self.leftNode = None
        self.rightNode = None
        self.gg = None
        self.gi = None
        self.depth = 0
        self.label = None


def build_tree(node):
#    print(node.depth,node.name)
    dataset = repeat(node.val)
    
    if dataset[2]==0:
        node.val=majority_vote(node.val)
        node.name = dataset[-1]

    elif node.depth==(max_depth-1):
        t1 = repeat(node.val)
        leaf_value = maj_vote(node.val,t1[-1])
        node.name = t1[-1]
        node.leftNode = TreeNode()
        node.rightNode = TreeNode()
        node.rightNode.depth = node.depth+1
        node.leftNode.depth = node.depth+1        
        node.leftNode.val = leaf_value[0]
        node.rightNode.val = leaf_value[1]

    elif node.depth<(max_depth-1):
        node.name = dataset[-1]
#         print(node.name)
        node.gg = dataset[2]
##         print(node.depth)
        
        if len(node.val) > 2:
#             print(node.val)
            node.leftNode = TreeNode()
            node.leftNode.depth = node.depth+1
            node.leftNode.val = dataset[0]
            build_tree(node.leftNode)
##             print(node.leftNode.depth)

        elif len(node.val) == 2:
##             print(node.name)
            var_l = list(node.val.keys())[0]
##             print(node.val)
            leaf_value = maj_vote(node.val,var_l)
##             print(var_l)
            node.leftNode = TreeNode()
            node.leftNode.name =[dataset[-1]]
##             node.rightNode = TreeNode()
##             node.rightNode.depth = node.depth + 1
            node.leftNode.depth = node.depth + 1   
            node.leftNode.val = leaf_value[0]


        if len(node.val) > 2:
            node.rightNode = TreeNode()
            node.rightNode.depth = node.depth+1            
            node.rightNode.val = dataset[1]
            build_tree(node.rightNode)
##             print(node.rightNode.depth)

        elif len(node.val) == 2:
##             print(node.name)
            var_r = list(node.val.keys())[0]
            leaf_value = maj_vote(node.val,var_r)
##             print(var_r)
##             node.leftNode = TreeNode()
            node.rightNode = TreeNode()
            node.rightNode.name = [dataset[-1]]
            node.rightNode.depth = node.depth+1

            node.rightNode.val = leaf_value[1]


def final_count(data):
    attribute = attributes(data)
    counter = counts(data,attribute)
    counter = list(counter.values())
    attribute = list(attribute.values())
    return counter+attribute

#def print_tree(node):
#    display = list()
#    if type(node) == TreeNode:
#        if len(node.val)>2:
#            print(counts(node.val,attributes(node.val)))
#            print((node.depth+1)*'|',node.name)
#            print_tree(node.leftNode)
##         print((node.depth+1)*'|',node.name)
##         print_tree(node.rightNode)    
            

def prep(data):
    l = list()
    for i in range(len(list(data.values())[0])):
        dic = dict()
        for item in data:
            if not item in dic:
                dic.update({item:data[item][i]})

        l.append(dic)
    return l

def prediction(train_data,attribute,node):
    class_feat = attribute[list(attribute.keys())[-1]]
#    print(node.name)
    if node.val in class_feat:
        return node.val
    else:
        if train_data[node.name] == attribute[node.name][0]:
            res = prediction(train_data,attribute,node.leftNode)
        else:
            res = prediction(train_data,attribute,node.rightNode)
            
        return res

def error(data,predicted):
    err = 0
    class_result = list(data.keys())[-1]
    for i in range(len(predicted)):
        if not data[class_result][i]==predicted[i]:
            err += 1
    err=err/len(predicted)
    return err

def main():    
    data = readfile('education_train.tsv')
    attribute = attributes(data)
    if max_depth == 0:
        result = majority_vote(data)
        print(result)
    
    else:
        root = TreeNode()
        root.val = data


        build_tree(root)
        
        train_data = prep(data)
        predicted = list()
        for i in range(len(train_data)):
            predicted.append(prediction(train_data[i],attribute,root))
        #         print_tree(root)
        print(predicted)
        
        faulty = error(data,predicted)
        print(faulty)
if __name__ == '__main__':
    main()