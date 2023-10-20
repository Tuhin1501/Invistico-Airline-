# Invistico-Airline-
transformed data, conducted statistical analysis and applied machine learning and cross validation techniques to find best model. Best model was then used to extract most relevant features affecting customer satisfaction for airline
data = pd.read_csv(r"C:\Users\t1u5h\Downloads\Invistico_Airline.csv")
data.isna().sum()
data.dropna(inplace=True)
data.info()
data.describe()
data["satisfaction"]=data["satisfaction"].map({"satisfied":1, "dissatisfied":0})
data["Class"] = data["Class"].map({"Business":3, "Eco Plus":2, "Eco":1})
data = pd.get_dummies(data, columns=["Type of Travel", "Customer Type"])
data_new = data.reset_index(drop=True)
y = data_new[["satisfaction"]]
x = data_new.drop(columns="satisfaction")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

xgb = XGBClassifier(objective='binary:logistic', random_state=0)
cv_params = {'max_depth': [4, 6],
              'min_child_weight': [3, 5],
              'learning_rate': [0.1, 0.2, 0.3],
              'n_estimators': [5,10,15],
              'subsample': [0.7],
              'colsample_bytree': [0.7]
              }

scoring = {"precision", "recall", "f1", "accuracy"}
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit="f1")
xgb_cv.fit(x_train,y_train)
xgb_cv.best_score_
xgb_cv.best_estimator_
def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.
  
    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.  
    '''

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = table.append({'Model': model_name,
                        'F1': f1,
                        'Recall': recall,
                        'Precision': precision,
                        'Accuracy': accuracy
                        },
                        ignore_index=True
                       )
  
    return table

y_pred = xgb_cv.predict(x_test)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
f1_score(y_test,y_pred)
recall_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred,labels=xgb_cv.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = xgb_cv.classes_)
disp.plot()

plot_importance(xgb_cv.best_estimator_)
