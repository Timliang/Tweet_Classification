from collections import Counter
import csv
import re
import nltk
import sklearn
import numpy
import matplotlib 

#get training data, testing data, and key attribute list.
def get_data(l1, l2):
    train_data = list(csv.reader(open(l1)))
    test_data = list(csv.reader(open(l2)))

    train_data = split_data(train_data)#delete punctuation and unless words.
    test_data = split_data(test_data)
    
    words = Counter() #occurrence number of each wrod in training data
    for i in range(len(train_data)):
        words.update(train_data[i][1])
    key_words = words.most_common(200)
    #pick most common words out first, then select more important ones out of them by other feature reduction method below.


    #0)select 10 most common elements for each category (50 features)
    #accuracy = [0.6619217081850534, 0.7046263345195729]
#    key_words = list()
#    for i in range(len(C1)):
#        train_data_sub = sub_table(train_data, C1[i],2)[0]
#        words = Counter() #all wrods in train_data
#        for i in range(len(train_data_sub)):
#            words.update(train_data[i][1])
#        key_words = key_words + words.most_common(10)
   
    #1)select key_words by removing features with low variance (30 features)
    #accuracy = [0.7402135231316725, 0.7615658362989324]
#    key_prob_table = get_key_prob_table(key_words, train_data)
#    sel = sklearn.feature_selection.VarianceThreshold(threshold=(.7 * (1 - .7)))
#    sel.fit_transform(key_prob_table)
#    key_prob_table = numpy.array(key_prob_table)
#    score = sorted(sel.variances_)
#    score.reverse()
#    model1_scores = sel.variances_.tolist()
#    key = list(model1_scores.index(score[i]) for i in range(30))
#    key_words = list(key_words[i] for i in key)
    
    #2)select key_words with highest p-values for chi-square (30 features)
    #accuracy = [0.7544483985765125, 0.7580071174377224]
#    key_prob_table = get_key_prob_table(key_words, train_data)
#    model1 = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=30)
#    model1.fit_transform(key_prob_table, list(c[2] for c in train_data))    
#    key_prob_table = numpy.array(key_prob_table)
#    score = sorted(model1.scores_)
#    score.reverse()
#    model1_scores = model1.scores_.tolist()
#    key = list(model1_scores.index(score[i]) for i in range(200))
#    key_words = list(key_words[i] for i in key)
    
    #3)select key_words by univariate feature selection (30 features)
    # why all three result is same as 2) ?????????
    #accuracy = [0.7544483985765125, 0.7580071174377224]
#    key_prob_table = get_key_prob_table(key_words, train_data)
#    #model1 = sklearn.feature_selection.SelectPercentile(score_func = sklearn.feature_selection.f_classif, percentile=15)
#    #model1 = sklearn.feature_selection.SelectFpr(score_func=sklearn.feature_selection.f_classif, alpha=0.05)
#    model1 = sklearn.feature_selection.GenericUnivariateSelect(score_func=sklearn.feature_selection.f_classif, mode='percentile', param=1e-05)
#
#    model1.fit_transform(key_prob_table, list(c[2] for c in train_data)) 
#
#    key_prob_table = numpy.array(key_prob_table)
#    score = sorted(model1.scores_)
#    score.reverse()
#    model1_scores = model1.scores_.tolist()
#    key = list(model1_scores.index(score[i]) for i in range(30))
#    key_words = list(key_words[i] for i in key)
    
    #4)Recursive feature elimination(RFE) 
    #accuracy = [0.7651245551601423, 0.7544483985765125] for (30 features)
    #accuracy = [0.7722419928825622, 0.7544483985765125] for (40 features)
    key_prob_table = get_key_prob_table(key_words, train_data)    
    key_prob_table = numpy.array(key_prob_table)
    X = key_prob_table.reshape((len(key_prob_table), -1))
    y = list(c[2] for c in train_data)
    
     #Create the RFE object and rank each pixel
    svc = sklearn.svm.SVC(kernel="linear", C=1)
    rfe = sklearn.feature_selection.RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    
    ranking = rfe.ranking_.reshape(key_prob_table[0].shape)
    score = sorted(ranking)
    model1_scores = ranking.tolist()
    key = list(model1_scores.index(score[i]) for i in range(40))
    key_words = list(key_words[i] for i in key)

    return train_data, test_data, key_words

# for each instance in training dataset, create a vector of indicator variable.
def get_key_prob_table(key_words, train_data):
    key_prob_table = list()
    for i in range(len(train_data)):
        key_prob_table.append([0]*len(key_words))
    for i in range(len(key_words)):
        for j in range(len(train_data)):
            if (key_words[i][0] in train_data[j][1]): 
                key_prob_table[j][i] = 1
    return key_prob_table

#split whole message into words.
def split_data(data):
    for i in range(len(data)): 
        data[i].append(data[i][1])
        normal = normalizer(data[i][1])
        tags = Counter()
        for word in normal:
            tags[word] += 1
        for dic_key in tags:
            tags[dic_key] = 1
        data[i][1] = tags
    return data

# delete unpopular words in english, delete punctuation.
def normalizer(tweet):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    only_letters = re.sub("[^a-zA-Z]", " ", tweet.lower()) 
    tokens = nltk.word_tokenize(only_letters)
    filtered_result = list(filter(lambda l: l not in stop_words, tokens))
    return filtered_result

# obtain sub-table containing only yi.
def sub_table(data, yi, k):
    new_data = list()
    for i in range(len(data)):
        if (data[i][k]==yi):
            new_data.append(data[i])
            
    all_words = Counter()
    for k in range(len(new_data)):
        all_words.update(new_data[k][1])
    return new_data, all_words

# package prior probability & conditional probability into single lists.
def probability_tables(C1, C2, key_words, data):
    prob = list() #prob[0]=prior, prior[1]=base on C1, prior[2]=base on C2
    for i in range(len(key_words)):
        prob.append([float(key_words[i][1])/len(data), conditional_prob(C1, key_words[i],
                     data, 2), conditional_prob(C2, key_words[i], data, 3)])
    
    p1 = list(); p2 = list()
    for c in C1: p1.append(float(len(sub_table(data, c, 2)[0]))/float(len(data)))
    for c in C2: p2.append(float(len(sub_table(data, c, 3)[0]))/float(len(data)))
    return prob, [p1, p2]

#calculate conditional probability.
def conditional_prob(C, key, data, k):
    p = list()
    for c in C:
        new_data, words = sub_table(data, c, k)
        a = words.get(key[0])
        if (a==None): a = 0.01 #2)default
        p.append(float(a)/float(len(new_data))) 
    return p

#find wether elements in or not in testing data
def elements_in(key_words, data):
    ind = list()#get a list of indicator variable,
    for i in range(len(data)):
        ind_i = list()
        for key in key_words:
            if(data[i][1].get(key[0])==None): ind_i.append(0)
            else: ind_i.append(1)
        ind.append(ind_i)
    return ind

#calculate posterioe probability for C1/C2
def posterior(prob_x, prob_y, data, key_words):
    ind = elements_in(key_words, data)
    post = list()
    for k in range(len(data)):
        p1 = list(); p2 = list() #C1;C2
        for i in range(len(prob_y[0])):
            p = prob_y[0][i]
            for j in range(len(ind[k])):
                if (ind[k][j]==1): p *= prob_x[j][1][i]/prob_x[j][0]
                else: p *= (1-prob_x[j][1][i])/(1-prob_x[j][0])
            p1.append(p)
        for i in range(len(prob_y[1])):
            p = prob_y[1][i]
            for j in range(len(ind[k])):
                if (ind[k][j]==1): p *= prob_x[j][2][i]/prob_x[j][0]
                else: p *= (1-prob_x[j][2][i])/(1-prob_x[j][0])
            p2.append(p)
        post.append([p1, p2])
    return post

# make prediction forC1/C2.
def predic(post, C1, C2):
    pr = list()
    for i in range(len(post)):
        c1 = post[i][0].index(max(post[i][0]))
        c2 = post[i][1].index(max(post[i][1]))               
        pr.append([C1[c1], C2[c2]])
    return pr

# calculate confusion matrix.
def accuracy(C1, C2, pred, test):
    confusion1 = list(); confusion2 = list()
    for i in range(len(C1)): confusion1.append([0]*len(C1))
    for i in range(len(C2)): confusion2.append([0]*len(C2))
    for i in range(len(test)):
        confusion1[C1.index(pred[i][0])][C1.index(test[i][2])] += 1
        confusion2[C2.index(pred[i][1])][C2.index(test[i][3])] += 1
    return confusion1, confusion2

# make plot of accuracy & number of attributes, which helps to select attribute number.
def cross_valid(train_data, key_words, C1, C2, N):
    accuracy_train = [0, 0]
    average_accuracy = []
    n = 0 #initialization
    for m in range(N):
        key_words_i = key_words[:m]
        accuracy_train_i = [0, 0]
        for i in range(10): #seperate data into ten parts
            l = len(train_data)/10
            train_data_1 = [train_data[i*l+j] for j in range(l)]
            train_data_2 = train_data[:i*l] + train_data[i*(l+1):]
            prob, prob_y= probability_tables(C1, C2, key_words_i, train_data_2)
            post = posterior(prob, prob_y, train_data_1, key_words_i)
            pred = predic(post, C1, C2)
            confusion1, confusion2 = accuracy(C1, C2, pred, train_data_1)
            accuracy_train_i[0] += float(sum([confusion1[i][i] for i in range(len(C1))]))/float(len(train_data_1))
            accuracy_train_i[1] += float(sum([confusion2[i][i] for i in range(len(C2))]))/float(len(train_data_1))
        if (((float(accuracy_train_i[0])/float(10) + float(accuracy_train_i[1])/float(10))/float(2) > float(accuracy_train[0] + accuracy_train[1])/float(2))):
            accuracy_train[0], accuracy_train[1] = float(accuracy_train_i[0])/float(10), float(accuracy_train_i[1])/float(10)
            n=m
        average_accuracy.append((float(accuracy_train_i[0])/float(10) + float(accuracy_train_i[1])/float(10))/float(2))
    matplotlib.pyplot.figure()  
    matplotlib.pyplot.plot(range(N),average_accuracy) 
    matplotlib.pyplot.savefig("easyplot.jpg") 
    return n

# Accuracy & output------------------------------------------------
def performance(confusion):
    s = 0; s_y = list([0]*len(confusion)); s_x = list([0]*len(confusion))
    acc = 0; acc_sub = list(); spe = list()
    rec = list(); pre = list(); f_1 = list()
    f_b_1 = list() #beta=0.5
    f_b_2 = list() #beta=2
    
    for i in range(len(confusion)):
        acc += float(confusion[i][i])
        s += sum(confusion[i])
        for j in range(len(confusion)):
            s_y[j] +=  confusion[i][j]
            s_x[i] +=  confusion[i][j]            
    for i in range(len(confusion)):
        if (s_y[i]==0):
            s_y[i]=1
        if (s_x[i]==0):
            s_x[i]=1        
    for i in range(len(confusion)):
        acc_sub.append(float(confusion[i][i]) / float(s))
        m = s - (s_x[i])
        if (m==0): m=1
        spe.append(float(acc-confusion[i][i]) / float(m))
        m = s_y[i]
        if (m==0): m=1
        pre.append(float(confusion[i][i])/float(m))
        m = s_x[i]
        if (m==0): m=1
        rec.append(float(confusion[i][i])/float(m))
        m = pre[i]+rec[i]
        if (m==0): m=1
        f_1.append(float(2*pre[i]*rec[i])/float(m))
        m = (0.5*0.5)*pre[i]+rec[i]
        if (m==0): m=1
        f_b_1.append(float((0.5*0.5 +1)*pre[i]*rec[i])/
                   float(m))
        m = (2*2)*pre[i]+rec[i]
        if (m==0): m=1
        f_b_2.append(float((2*2 +1)*pre[i]*rec[i])/
                   float(m))
    acc = round(float(acc)/float(s), 2)
    for i in range(len(confusion)):
        acc_sub[i] = round(acc_sub[i], 2)
        spe[i] = round(spe[i], 2)
        pre[i] = round(pre[i], 2)
        rec[i] = round(rec[i], 2)
        f_1[i] = round(f_1[i], 2)
        f_b_1[i] = round(f_b_1[i], 2)
        f_b_2[i] = round(f_b_2[i], 2)
    return [acc, pre, rec, f_1]
 
def print_matrix(matrix):
    for i in range(len(matrix)):
        print '\n'
        for j in range(len(matrix)):
            print matrix[i][j], '\t',
    print '\n'
      
if __name__ == "__main__": 
    train_data, test_data, key_words = get_data("/Users/xiaowen/Desktop/labeled-data-singlelabels-train.csv", "/Users/xiaowen/Desktop/labeled-data-singlelabels-test.csv")
    C1 = ["Energy", "Food", "Medical", "None", "Water"]
    C2 = ["N/A", "need", "resource"]    
    prob, prob_y= probability_tables(C1, C2, key_words, train_data)

     #accuracy for train_data
#    post = posterior(prob, prob_y, train_data, key_words)
#    pred = predic(post, C1, C2)
#    confusion1, confusion2 = accuracy(C1, C2, pred, train_data)
#    accuracy_train = [float(sum([confusion1[i][i] for i in range(len(C1))]))/float(len(train_data)), float(sum([confusion2[i][i] for i in range(len(C2))]))/float(len(train_data))]

     #accuracy for test_data
    post = posterior(prob, prob_y, test_data, key_words)
    pred = predic(post, C1, C2)
    confusion1, confusion2 = accuracy(C1, C2, pred, test_data)
    output = [performance(confusion1), performance(confusion2)]   
    print "confusion matrix of category1 test:"
    print_matrix(confusion1)
    print "confusion matrix of category2 test:"
    print_matrix(confusion2)
    
#    accuracy_test = [float(sum([confusion1[i][i] for i in range(len(C1))]))/float(len(test_data)), float(sum([confusion2[i][i] for i in range(len(C2))]))/float(len(test_data))]
#    cross_valid(train_data, key_words, C1, C2, 3)
