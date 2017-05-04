import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def load_prepare_dataset():
	reservations = pd.read_csv('reservations_(5).csv')
	vehicles = pd.read_csv('vehicles_(6).csv') 
	
	reservations = reservations.pivot_table(index = 'vehicle_id',columns = 'reservation_type',aggfunc = len, fill_value = 0)
	#merging datasets 
	data = pd.merge(vehicles, reservations,left_on = 'vehicle_id',right_index = True,how = 'outer',sort = False) 
	#filling missing observations with 0 
	data.fillna(value = 0, inplace = True) 
	#renaming columns 
	data.rename(columns = {1:'hourly',2:'daily',3:'weekly'},inplace = True)
	#computing total rentals for a vehicle_id 
	data['total_rentals'] = data['hourly']+data['daily']+data['weekly']
	return data 

def box_plot(data,x,y):
	data.boxplot(y, by = x)
	plt.show() 

def scatter_plot(x,y,x_label,y_label):
	plt.scatter(x,y)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show() 

#Linear regression 
def linear_regression(x,y):
	x2 = sm.add_constant(x)
	est = sm.OLS(y,x2)
	est2 = est.fit()
	print(est2.summary())
	mean_squared_error(y,  est2.predict(x2))
	
def ridge_regression(x,y):
    alphas = np.linspace(0.01,100,1000)
    rdg = RidgeCV(alphas = (alphas), fit_intercept=True, normalize = True, scoring = 'r2', cv=None, gcv_mode = None,store_cv_values =False)
    model = rdg.fit(x, y)
    print("alpha of ridge regression: ", model.alpha_) #0.01 
    print("coefficients of ridge model: ",model.coef_) 
    print("R2: ridge ", model.score(x,y)) #0.199 with old features 
    mean_squared_error(y,  rdg.predict(x))

def decision_tree(x,y):
    param_test1 = {'max_depth':range(3,22,2),'min_samples_split':range(2,20,2),'min_samples_leaf':range(1,20,2),'max_features':['auto','sqrt','log2']}
    gsearch1 = GridSearchCV(estimator = DecisionTreeRegressor(criterion ='mse', max_depth=5,min_samples_split=4, min_samples_leaf=2, random_state = 22), param_grid = param_test1, scoring='r2',n_jobs=1,iid=False)
    gsearch1.fit(x,y)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_ 

    #best dtree 
    dtreg = DecisionTreeRegressor(max_depth=5,min_samples_split=2, min_samples_leaf=19, random_state = 22)
    dtreg.fit(x,y)
    dtreg.feature_importances_#array([ 0.01929037,  0.48749129,  0.2800381 ,  0.18645132,  0.00549343,0.02123549]) 
    print("R2: decision tree ",dtreg.score(x,y)) #0.31758 - new data 
    mean_squared_error(y,  dtreg.predict(x))
	
    #plotting decision tree 
    features = ['technology','actual_price','recommended_price','num_images','street_parked','description'] 
    dotfile = open("C:/Users/deeps/python/Kaggle/Turo/decturo.dot",'w') 
    dotfile = tree.export_graphviz(dtreg, out_file = dotfile, feature_names = features)  #max_depth=9,min_samples_split=2, min_samples_leaf=19, random_state = 22 for new features

	
def random_forest(x,y):
    param_test1 = {'max_depth':range(3,22,2),'min_samples_split':range(2,20,2),'min_samples_leaf':range(1,20,2)}
    gsearch1 = GridSearchCV(estimator = RandomForestRegressor( n_estimators =500, criterion ='mse', max_depth=5,min_samples_split=4, min_samples_leaf=2, oob_score = True, random_state = 22), param_grid = param_test1, scoring='r2',n_jobs=1,iid=False)
    gsearch1.fit(x,y)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_ #new data - ({'max_depth': 5, 'min_samples_split': 16, 'min_samples_leaf': 7}, 0.23717457876740997) 

    #new data rf best 
    rfreg = RandomForestRegressor(n_estimators =500, criterion ='mse', max_depth=5,min_samples_split=16, min_samples_leaf=7, oob_score = True, random_state = 22)
    rfreg.fit(x,y)
    rfreg.feature_importances_ #array([ 0.00840439,  0.42210192,  0.33831274,  0.14825909,  0.0108987 , 0.07202316])
    rfreg.oob_score_ #0.222467 
    print("R2: random forest ", rfreg.score(x,y) #0.377258 #new data 
    mean_squared_error(y,  rfreg.predict(x)) #max_depth=9,min_samples_split=2, min_samples_leaf=7 for new features 
	
def gbm(x,y):
    param_test1 = {'learning_rate':np.linspace(0.01,0.3,10),'n_estimators':range(100,1500,200)}
    gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor( learning_rate=0.01,n_estimators =500, max_depth=5, min_samples_leaf=7, alpha = 0.1, random_state = 22), param_grid = param_test1, scoring= 'neg_mean_squared_error',n_jobs=1,iid=False)
    gsearch1.fit(x,y)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_ 
	
	#gbmreg best 
    gbmreg = GradientBoostingRegressor(learning_rate=0.01,  n_estimators = 500, max_depth = 5, min_samples_leaf = 7, alpha = 0.1)
    gbmreg.fit(x,y) 
    gbmreg.feature_importances_ #array([ 0.02758982,  0.44213884,  0.25794564,  0.08132225,  0.01504151,      0.17596195])
    gbmreg.score(x,y) #default gives 0.5, this one 0.58 but mse is just 9.7428 
	
    gbmreg = GradientBoostingRegressor(learning_rate=0.01,  n_estimators = 100, max_depth = 5, min_samples_leaf = 7, alpha = 0.1) 
    gbmreg.fit(x,y) 
    gbmreg.feature_importances_ #array([ 0.02290171,  0.4145783 ,  0.35004126,  0.12005878,  0.00661026,
 #       0.08580969])
    gbmreg.score(x,y) #0.3535 mse is 15.26 
	
data_ = load_prepare_dataset() 
print(data_.head(8)) 

#assuming underlying data is linear 
x = data_.drop(['vehicle_id','hourly','daily','weekly','total_rentals'],axis = 1)
y = data_['total_rentals'] 
linear_regression(x,y) #R2 - 0.569, adj-R2 - 0.565 with new features

#dropping street_parked and fitting linear regression model 
x2 = x.drop('street_parked', axis = 1) 
linear_regression(x2,y) #R2 - 0.569, adj-R2 - 0.565 with new features

#dropping description 
x3 = x2.drop('description', axis = 1) 
linear_regression(x3,y) #R2 - 0.569, adj-R2 - 0.566 with new features

#dropping technology 
x4 = x3.drop('technology', axis = 1) 
linear_regression(x4,y) #R2 - 0.564, adj-R2 - 0.561 with new features 

#ridge regression 
ridge_regression(x,y)

#decision tree 
decision_tree(x,y)

#random forest 
random_forest(x,y)

#gradient boosting 
gbm(x,y)


box_plot(data_,'num_images','total_rentals')
box_plot(data_,'street_parked','total_rentals')
box_plot(data_,'technology','total_rentals')
box_plot(data_,'technology','hourly')
box_plot(data_,'technology','daily')
box_plot(data_,'technology','weekly')

scatter_plot(data_.description,data_.total_rentals,'description_length','total_rentals')
scatter_plot(data_.actual_price, data_.total_rentals,'actual_price','total_rentals')
scatter_plot(data_.recommended_price, data_.total_rentals,'recommended_price','total_rentals')
scatter_plot(data_.recommended_price, data_.actual_price,'recommended_price','actual_price')






