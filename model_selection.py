import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LinearRegression
import sys; import re

def AIC(data,model,model_type,k=2):
	if model_type=='linear':
		return len(data)* np.log(model.ssr/len(data)) + k * (model.df_model+1)
	elif model_type=='logistic' :
		return model.aic


def Cp(data,model,sigma2):
	return model.ssr/sigma2 - (len(data) - 2.*model.df_model- 1)

def BIC(data,model,model_type='linear'):
	if model_type=='linear':
		return np.log(model.ssr/model.centered_tss) * len(data) + (model.df_model+1) * np.log(len(data))
	elif model_type=='logistic':
		return model.bicllf

def regressor(y,X, model_type):
	if model_type =="linear":
		regressor = sm.OLS(y, X)
		regressor_fitted = regressor.fit()

	elif model_type == 'logistic':
		regressor = sm.GLM(y, X,family=sm.families.Binomial())
		regressor_fitted = regressor.fit()
	return regressor_fitted



def criterion_f(X,model,model_type,elimination_criterion):
	if elimination_criterion=='aic':
		return AIC(X,model,model_type)
	elif elimination_criterion=='bic':
		return AIC(X,model,model_type,k=np.log(len(X)))



def detect_dummies(X,variable):
	'''
	If no dummies simply returns the variable to remove (or add)
	'''
	cols = X.columns.tolist()

	dummy_cols = []
	if (len(X[variable].value_counts())==2) and (X[variable].min()==0) and (X[variable].max()==1):
		cols.remove(variable)
		dummy_cols.append(variable)

		if re.search('^([a-zA-Z0-9]+)[\[_]',variable):
			prefix = (re.search('^([a-zA-Z0-9]+)[\[_]',variable).group(1))
			for var in cols:
				if prefix in var:
					dummy_cols.append(var)
	else :
		dummy_cols.append(variable)
	return dummy_cols

def forwardSelection(X, y, model_type ="linear",elimination_criterion = "aic",verbose=False):
	'''
	Compute model selection based on elimination_criterion (here only aic and bic)
	Forward Selection : from simple model with only intercept to complete model with all variables present in X
	X : predictors, pandas dataframe nxp or array nxp
	y : output, pandas series (DataFrame with one column), nx1, or 1d array of length n
	elimination_criterion : here only aic available
	----
	returns final model fitted with selected variables
	'''
	return __forwardSelectionRaw__(X, y, model_type = model_type,elimination_criterion = elimination_criterion,verbose=verbose)

def backwardSelection(X, y, model_type ="linear",elimination_criterion = "aic",verbose=False):
	'''
	Compute model selection based on elimination_criterion (here only aic and bic)
	Backward Selection : from complete with all columns in X to simple model with only intercept
	X : predictors, pandas dataframe nxp or array nxp
	y : output, pandas series (DataFrame with one column), nx1, or 1d array of length n
	elimination_criterion : here only aic available
	----
	returns final model fitted with selected variables
	'''
	return __backwardSelectionRaw__(X, y, model_type = model_type,elimination_criterion = elimination_criterion,verbose=verbose )


def bothSelection(X, y, model_type ="linear",elimination_criterion = "aic",start='full',verbose=False):
	return __bothSelectionRaw__(X, y, model_type = model_type,elimination_criterion = elimination_criterion,start=start,verbose=verbose)



def __forwardSelectionRaw__(X, y, model_type ="linear",elimination_criterion = "aic",verbose=False):

	cols = X.columns.tolist()

	## Begin from a simple model with only intercept
	selected_cols = ["Intercept"]
	other_cols = cols.copy()
	other_cols.remove("Intercept")

	model = regressor(y, X[selected_cols],model_type)
	criterion = criterion_f(X,model,model_type,elimination_criterion)


	for i in range(X.shape[1]):
		aicvals = pd.DataFrame(columns = ["Cols","aic"])
		for j in other_cols:
			cols_to_add = detect_dummies(X,j)
			model = regressor(y, X[selected_cols+cols_to_add],model_type)
			aicvals = aicvals.append(pd.DataFrame([[j, criterion_f(X,model,model_type,elimination_criterion)]],columns = ["Cols","aic"]),ignore_index=True)

		aicvals = aicvals.sort_values(by = ["aic"]).reset_index(drop=True)
		if verbose :
			print(aicvals)
		if aicvals.shape[0] > 0:
			new_criterion = aicvals["aic"][0]
			if new_criterion < criterion:
				cols_to_add = detect_dummies(X,aicvals["Cols"][0])
				print("Entered :", aicvals["Cols"][0], "\tCriterion :", aicvals["aic"][0])
				for i in cols_to_add:
					selected_cols.append(i)
					other_cols.remove(i)
				criterion = new_criterion
			else:
				print("break : criterion")
				break


	model = regressor(y,X[selected_cols],model_type)

	print(model.summary())
	print("Criterion: "+str(criterion_f(X,model,model_type,elimination_criterion)))
	print("Final Variables:", selected_cols)

	return model

def __backwardSelectionRaw__(X, y, model_type ="linear",elimination_criterion = "aic",verbose=False):


	selected_cols = X.columns.tolist()
	selected_cols.remove('Intercept')

	model = regressor(y,X,model_type)
	criterion = criterion_f(X,model,model_type,elimination_criterion)

	for i in range(X.shape[1]):

		aicvals = pd.DataFrame(columns = ["Cols","aic"])
		if len(selected_cols)==0:
			print("break : Only Intercept left")
			break
		else :
			for j in selected_cols:
				temp_cols = selected_cols.copy()
				### Detect dummies and remove several columns if necessary
				cols_to_remove = detect_dummies(X,j)

				for i in cols_to_remove:
					temp_cols.remove(i)

				model = regressor(y, X[['Intercept']+temp_cols],model_type)
				aicvals = aicvals.append(pd.DataFrame([[j, criterion_f(X,model,model_type,elimination_criterion)]],columns = ["Cols","aic"]),ignore_index=True)

			aicvals = aicvals.sort_values(by = ["aic"]).reset_index(drop=True)

			if verbose :
				print(aicvals)
			new_criterion = aicvals["aic"][0]
			if new_criterion < criterion:
				print("Eliminated :" ,aicvals["Cols"][0],"\tCriterion :", aicvals["aic"][0])
				cols_removed = detect_dummies(X,aicvals["Cols"][0])

				for i in cols_removed:
					selected_cols.remove(i)
				criterion = new_criterion
			else:
				print("break : criterion")
				break


	model = regressor(y,X[['Intercept']+selected_cols],model_type)

	print(str(model.summary())+"\nCriterion: "+ str(criterion_f(X,model,model_type,elimination_criterion)))
	print("Final Variables:", selected_cols)


	return model

def __bothSelectionRaw__(X, y, model_type ="linear",elimination_criterion = "aic",start='full',verbose=False):
	'''
	Compute model selection based on elimination_criterion (here only aic and bic)
	Both direction Selection : from complete (full) with all columns in X to simple model with only intercept, but try to add or delete one variable at each step
	X : predictors, pandas dataframe nxp or array nxp
	y : output, pandas series (DataFrame with one column), nx1, or 1d array of length n
	elimination_criterion : here only aic available
	----
	returns final model fitted with selected variables
	'''

	cols = X.columns.tolist()

	if start=='full':
		removed_cols = []
		selected_cols = cols.copy()
		selected_cols.remove("Intercept")

	else :
		selected_cols = []
		removed_cols = cols.copy()
		removed_cols.remove("Intercept")


	model = regressor(y,X[['Intercept']+selected_cols],model_type)
	criterion = criterion_f(X,model,model_type,elimination_criterion)

	while True :
		aicvals = pd.DataFrame(columns = ["Cols","aic",'way'])

		###### Try to remove variables still present in the model
		if len(selected_cols)==0:
			continue
		else :
			for j in selected_cols:
				temp_cols = selected_cols.copy()

				### Detect dummies and remove several columns if necessary
				cols_to_remove = detect_dummies(X,j)
				for i in cols_to_remove:
					temp_cols.remove(i)

				model = regressor(y, X[['Intercept']+temp_cols],model_type)
				aicvals = aicvals.append(pd.DataFrame([[j, criterion_f(X,model,model_type,elimination_criterion),'delete']],columns = ["Cols","aic",'way']),ignore_index=True)

		###### Try to add previously removed variables
		for j in removed_cols:
			cols_to_add = detect_dummies(X,j)

			model = regressor(y, X[['Intercept']+selected_cols+cols_to_add],model_type)
			aicvals = aicvals.append(pd.DataFrame([[j, criterion_f(X,model,model_type,elimination_criterion),'add']],columns = ["Cols","aic",'way']),ignore_index=True)

		aicvals = aicvals.sort_values(by = ["aic"]).reset_index(drop=True)

		if verbose :
			print(aicvals)
		if aicvals.shape[0] > 0:
			new_criterion = aicvals["aic"][0]
			if new_criterion < criterion:
				cols_concerned = detect_dummies(X,aicvals["Cols"][0])

				if aicvals["way"][0]=='delete':
					print("Eliminated :" ,aicvals["Cols"][0],"\tCriterion :", aicvals["aic"][0])
					criterion = new_criterion
					for i in cols_concerned:
						selected_cols.remove(i)
						removed_cols.append(i)
					# removed_cols.append(aicvals["Cols"][0])
					# selected_cols.remove(aicvals["Cols"][0])
				elif aicvals["way"][0]=='add':
					print("Entered :", aicvals["Cols"][0], "\tCriterion :", aicvals["aic"][0])

					for i in cols_concerned:
						selected_cols.append(i)
						removed_cols.remove(i)
					# selected_cols.append(aicvals["Cols"][0])
					# removed_cols.remove(aicvals["Cols"][0])
					criterion = new_criterion
			else:
				print("break : criterion")
				break


	model = regressor(y,X[['Intercept']+selected_cols],model_type)
	print(str(model.summary())+"\nCriterion: "+ str(criterion_f(X,model,model_type,elimination_criterion)))
	print("Final Variables:", selected_cols)



	return model


def exhaustivesearch_selectionmodel(X,y,vmin=1,vmax=10):

	'''
	Function to compute exhaustive search for LINEAR regression y ~X  : test all models with p features from X  with p between vmin and vmax.
	For each size p  : select the best model based on MSE.
	Then compute R2,adj R2, Cp and BIC on selected models.
	X : Dataframe of explanatory variables, WITHOUT intercept column, nxp
	y : Dataframe of output variable
	---------
	Returns these different criterion in a DataFrame.

	'''

	if ('const' in X.columns.tolist()) or ('Intercept' in X.columns.tolist()):
		raise SystemExit('Delete Intercept column in X before to pass it to this function')
		# sys.exit('Delete Intercept column in X before to pass it to this function')

	### First, exhaustive search with LienarRegression() from sklearn and EFS() from mlxtend
	### Returns a dictionnary with all estimated models for each model dimension
	lm = LinearRegression(fit_intercept=True)
	efs1 = EFS(lm,min_features=1,max_features=vmax,scoring='neg_mean_squared_error',print_progress=True,cv=False)
	efs1 = efs1.fit(X, y)


	#### Find for each model size the best model in terms of (neg) MSE
	best_idxs_all = []
	for k in range(1,vmax+1):
		best_score = -np.infty
		best_idx = 0
		for i in efs1.subsets_:
			if (len(efs1.subsets_[i]['feature_idx'])) == k:
				if efs1.subsets_[i]['avg_score'] > best_score:
					best_score = efs1.subsets_[i]['avg_score']
					best_idx = i

		best_idxs_all.append(best_idx)


	df_subsets = pd.DataFrame(index=best_idxs_all,columns=['Variables','R2','R2_adj','Cp','BIC','Number of variables (except intercept)'])
	X_copy = X.copy()
	X_copy = sm.add_constant(X_copy)
	full_model = sm.OLS(y,X_copy).fit()
	sigma2 = (full_model.ssr)/(len(X_copy)-full_model.df_model-1)


	for index in best_idxs_all:
		df_subsets['Variables'] =  df_subsets['Variables'].astype(object)
		variables = (efs1.subsets_[index]['feature_names'])
		variables = np.array(variables).tolist()
		df_subsets.loc[index,'Number of variables (except intercept)'] = len(variables)
		model = sm.OLS(y,X_copy[['const']+variables]).fit()
		df_subsets.loc[index,'R2'] = model.rsquared
		df_subsets.loc[index,'R2_adj'] = model.rsquared_adj
		df_subsets.loc[index,'BIC'] = BIC(X_copy,model)
		df_subsets.loc[index,'Cp'] = Cp(X_copy,model,sigma2)
		df_subsets.loc[index,'Variables'] = variables




	return df_subsets
