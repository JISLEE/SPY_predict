# AI539: MACHINE LEARNING CHALLENGES (Final implementation)
# SPY(S&P 500 ETF) price trends prediction
# Author: Jisoo Lee
# date: Mar 14, 2022


import timeit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


def Classifiers(X_train, y_train, X_val, y_val, X_test, y_test, tag = 'linear'):
    
    # Dummy classifier: Most frequent
    start_time = timeit.default_timer()
    dummy_model = DummyClassifier(strategy='most_frequent', random_state=100).fit(X_train, y_train)
    terminate_time = timeit.default_timer()

    d_pred_val = dummy_model.predict(X_val)
    d_acc_val = accuracy_score(y_val, d_pred_val)

    d_pred_test = dummy_model.predict(X_test)
    d_acc_test = accuracy_score(y_test, d_pred_test)
    
    d_time = terminate_time - start_time
#     print("[Dummy] Training time: %f, Val acc: %f, Test acc: %f" % (d_time, d_acc_val, d_acc_test)) 


    # Gradient Boosting Classifier

    start_time = timeit.default_timer()
    gradient_boosting = GradientBoostingClassifier(learning_rate=0.001, n_estimators=100, random_state=100).fit(X_train, y_train)
    terminate_time = timeit.default_timer()

    gb_y_pred_val = gradient_boosting.predict(X_val)
    gb_acc_val = accuracy_score(y_val, gb_y_pred_val)

    gb_pred_test = gradient_boosting.predict(X_test)
    gb_acc_test = accuracy_score(y_test, gb_pred_test)

    gb_time = terminate_time - start_time
#     print("[Gradient Boosting] Training time: %f, Val acc: %f, Test acc: %f" % (gb_time, gb_acc_val, gb_acc_test))


    # Logistic Regression

    start_time = timeit.default_timer()
    logistic_model = LogisticRegression(penalty='l2', random_state = 100,max_iter=100).fit(X_train, y_train)
    terminate_time = timeit.default_timer()

    log_y_pred_val = logistic_model.predict(X_val)
    log_acc_val = accuracy_score(y_val, log_y_pred_val)

    log_pred_test = logistic_model.predict(X_test)
    log_acc_test = accuracy_score(y_test, log_pred_test)
    
    log_time = terminate_time - start_time
#     print("[Logistic Regression] Training time: %f, Val acc: %f, Test acc: %f" % (log_time, log_acc_val, log_acc_test))


    # Random Forest Classifier

    start_time = timeit.default_timer()
    random_forest = RandomForestClassifier(max_depth=100, random_state=100).fit(X_train, y_train)
    rf_y_pred_val = random_forest.predict(X_val)
    terminate_time = timeit.default_timer()

    rf_acc_val = accuracy_score(y_val, rf_y_pred_val)

    rf_pred_test = random_forest.predict(X_test)
    rf_acc_test = accuracy_score(y_test, rf_pred_test)
    
    rf_time = terminate_time - start_time

#     print("[Random Forest] Training time: %f, Val acc: %f, Test acc: %f" % (rf_time, rf_acc_val, rf_acc_test))


    # Multi Layer Perceptron

    start_time = timeit.default_timer()
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', learning_rate_init=0.001, random_state=100, max_iter=1000).fit(X_train, y_train)
    terminate_time = timeit.default_timer()

    mlp_y_pred_val = mlp_model.predict(X_val)
    mlp_acc_val = accuracy_score(y_val, mlp_y_pred_val)

    mlp_pred_test = mlp_model.predict(X_test)
    mlp_acc_test = accuracy_score(y_test, mlp_pred_test)
    
    mlp_time = terminate_time - start_time

#     print("[MLP] Training time: %f, Val acc: %f, Test acc: %f" % (mlp_time, mlp_acc_val, mlp_acc_test)) 
    
    dict = {"Dummy(most_frequent)": [d_time, d_acc_val, d_acc_test],
            "Gradient boosting": [gb_time, gb_acc_val, gb_acc_test],
            "Logistic regression": [log_time, log_acc_val, log_acc_test],
            "Random forest": [rf_time, rf_acc_val, rf_acc_test],
            "Multi layer perceptron": [mlp_time, mlp_acc_val, mlp_acc_test]
            }
    
    if tag=='linear':
        print("***** Experiment Result (Linear interpolation) *****\n")
    elif tag=='ffill':
        print("***** Experiment Result (Forward filling) *****\n")
    elif tag=='avg':
        print("***** Experiment Result (Average filling) *****\n")

    print ("{:<22} {:<14} {:<14} {:<14} ".format('Classifier', 'training time', 'val acc','test acc'))
    for k, v in dict.items():
        time, val, test = v
        print ("{:<22} {:<14} {:<14} {:<14} ".format(k, round(time, 4), round(val, 4), round(test, 4)))

        
def ClsExplanation(X_train, y_train, X_val, y_val, X_test, y_test, feature):

    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', learning_rate_init=0.001, random_state=100, max_iter=1000).fit(X_train, y_train)

    mlp_pred_test = mlp_model.predict(X_test)

    updown = ""
    label = ""
    for j in range(1188, 1191):
        print("When ")
        for i in range(X_test.shape[1]):
            feat_name = feature[i]
            if X_test[feat_name][j]>0:
                updown = "up"
            else:
                updown = "down"
            print(feat_name, "is ", updown,",")
        if mlp_pred_test[j-1159]==1:
            label = "fall"
        else: 
            label = "rise"
        print("The model predicts SPY price will ", label)
        print("\n")
    
    
    
