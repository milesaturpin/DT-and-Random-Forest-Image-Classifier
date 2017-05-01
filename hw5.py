import numpy
from classifiers import create_decision_tree, create_random_forest, calculate_model_accuracy, calculate_confusion_matrix
from data import get_minecraft, get_first_n_samples

def p0(featuretype='histogram'):
    data_train, data_test, target_train, target_test = get_minecraft(featuretype)
    model = create_decision_tree()

    # TODO: Fit the model to the data using its fit method

    model = model.fit(data_train, target_train)

    # TODO: Use the model's predict method to predict labels for the training and test sets
    predict_train = model.predict(data_train)
    predict_test = model.predict(data_test)

    accuracy_train, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target_train, target_test)
    print('Training accuracy: {0:3f}, Test accuracy: {1:3f}'.format(accuracy_train, accuracy_test))

    cfm = calculate_confusion_matrix(predict_test,target_test)
    print "Confusion matrix"
    print cfm

    for q in range(1,3):
        for p in range(0,q):
            #compute confusion between classes p and q
            index_pq = [i for i,v in enumerate(target_train) if v in [p,q]]
            modelpq = create_decision_tree()
            #TODO: fit model to the data only involving classes p and q
            relevant_data_train = [data_train[c] for c in index_pq]
            relevant_target_train = [target_train[c] for c in index_pq]

            modelpq = model.fit(relevant_data_train, relevant_target_train)

            testindex_pq = [i for i,v in enumerate(target_test) if v in [p,q]]
            #TODO: calculate and print the accuracy
            
            relevant_result = [data_test[c] for c in testindex_pq]
            relevant_result2 = [target_test[c] for c in testindex_pq]

            accuracy_pq = modelpq.score(relevant_result,relevant_result2)
            print "One-vs-one accuracy between classes",p,"and",q,":",accuracy_pq

    return model, predict_train, predict_test, accuracy_train, accuracy_test


def p1():
    #TODO: compare different feature types



    m,ptrain,ptest,atrain,atest = p0('histogram')
    m1,ptrain1,ptest1,atrain1,atest1 = p0('rgb')
    m2,ptrain2,ptest2,atrain2,atest2 = p0('gray')

    print "Histogram"
    print "accuracy on training set:",atrain," ... accuracy on test set", atest
    print "RGB"
    print "accuracy on training set:",atrain1," ... accuracy on test set", atest1
    print "Gray"
    print "accuracy on training set:",atrain2," ... accuracy on test set", atest2

def p2():
    results = []
    model = create_decision_tree()

    # TODO: Get the Minecraft dataset using get_minecraft() and create a decision tree
    data_train, data_test, target_train, target_test = get_minecraft('histogram')


    for n in [50, 100, 150, 200, 250]:
        # TODO: Fit the model using a subset of the training data of size n
        # Hint: use the get_first_n_samples function imported from data.py
        adjusted_data, adjusted_target = get_first_n_samples(data_train, target_train, n)

        #TODO: use the model to fit the training data and predict labels for the training and test data
        model = model.fit(adjusted_data, adjusted_target)
        predict_train = model.predict(adjusted_data)
    	predict_test = model.predict(data_test)

        # TODO: Calculate the accuracys of the model (use the training data that fit the model in the current iteration)
        accuracy_train_n, accuracy_test = calculate_model_accuracy(predict_train, predict_test, adjusted_target, target_test)

        results.append((n, accuracy_train_n, accuracy_test))

    print(results)
    return model, results


def p3():
    results = []
    model = None

    # TODO: Get the Minecraft dataset
    data_train, data_test, target_train, target_test = get_minecraft('histogram')

    for n_estimators in [2, 5, 10, 20, 30]:
        # TODO: create a random forest classifier with n_estimators estimators
        model = create_random_forest(n_estimators)

        #TODO: use the model to fit the training data and predict labels for the training and test data
        model = model.fit(data_train, target_train)
        predict_train = model.predict(data_train)
    	predict_test = model.predict(data_test)


        # TODO: calculate the accuracies of the models and add them to the results
        accuracy_train, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target_train, target_test)

        results.append((n_estimators, accuracy_train, accuracy_test))

    print results
    return model, results


def bonus():
    results = []
    model = None

    # OPTIONAL: Repeat p0 using a logistic regression classifier
    

    return model, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", type=str, choices=['p0', 'p1', 'p2', 'p3', 'bonus'], help="The problem to run")
    args = parser.parse_args()

    if args.problem == 'p0':
        p0()
    elif args.problem == 'p1':
        p1()
    elif args.problem == 'p2':
        p2()
    elif args.problem == 'p3':
        p3()
    elif args.problem == 'bonus':
        bonus()
