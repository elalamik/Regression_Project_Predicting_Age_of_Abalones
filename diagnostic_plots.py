import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.graphics as smg
import pandas as pd



def plot(results):

	infl = results.get_influence()



	# fitted values (need a constant term for intercept)
	model_fitted_y = results.fittedvalues

	# model residuals
	model_residuals = results.resid

	# normalized residuals
	model_norm_residuals = infl.resid_studentized_internal

	# absolute squared normalized residuals
	model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

	# absolute residuals
	model_abs_resid = np.abs(model_residuals)

	# leverage, from statsmodels internals
	model_leverage = infl.hat_matrix_diag

	# cook's distance, from statsmodels internals
	model_cooks = infl.cooks_distance[0]

	indexes = model_fitted_y.index


	if np.isnan(model_cooks).any():
		index_null = np.argwhere(np.isnan(model_cooks))
		model_cooks = np.delete(model_cooks,index_null)
		model_leverage = np.delete(model_leverage,index_null)
		model_norm_residuals = np.delete(model_norm_residuals,index_null)


	fig,axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)

	############################
	#### Residual plot
	############################


	plot_lm_1 = axes[0,0]

	sns.residplot(x=model_fitted_y, y=results.model.endog, ax=plot_lm_1,
	                                  lowess=True,
	                                  scatter_kws={'alpha': 0.5},
	                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

	plot_lm_1.set_title('Residuals vs Fitted')
	plot_lm_1.set_xlabel('Fitted values')
	plot_lm_1.set_ylabel('Residuals')
	plot_lm_1.set_xlim(np.min(model_fitted_y)-0.1, np.max(model_fitted_y)+0.5)

	# annotations
	abs_resid = model_abs_resid.sort_values(ascending=False)
	abs_resid_top_3 = abs_resid[:3]

	for i in abs_resid_top_3.index:
	    plot_lm_1.annotate(i,xy=(model_fitted_y[i],model_residuals[i]))




	############################
	#### Q-Q plot
	############################
	plot_lm_2 = axes[0,1]


	sm.qqplot(model_norm_residuals,fit=True,line='q',ax=plot_lm_2,alpha=0.5,color='#4C72B0')
	QQ = smg.gofplots.ProbPlot(data=np.sort(model_norm_residuals),fit=True) #,dist='norm')


	plot_lm_2.set_title('Normal Q-Q')
	plot_lm_2.set_xlabel('Theoretical Quantiles')
	plot_lm_2.set_ylabel('Standardized Residuals')

	# annotations


	abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
	abs_norm_resid_top_3 = abs_norm_resid[:3]
	abs_norm_resid_top_3_qq = np.flip(np.argsort(np.abs(QQ.sorted_data)),0)[:3]

	for r, i in enumerate(abs_norm_resid_top_3_qq):
		plot_lm_2.annotate(indexes[abs_norm_resid[r]],xy=(QQ.theoretical_quantiles[i],QQ.sorted_data[i]))


	############################
	#### Scale-location plot
	############################

	plot_lm_3 = axes[1,0]

	plot_lm_3.scatter(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, alpha=0.5)
	sns.regplot(x=model_fitted_y, y=model_norm_residuals_abs_sqrt,
	            scatter=False,
	            ci=False,
	            lowess=True,ax=plot_lm_3,
	            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

	plot_lm_3.set_title('Scale-Location')
	plot_lm_3.set_xlabel('Fitted values')
	plot_lm_3.set_ylabel('$\sqrt{|Standardized Residuals|}$');
	plot_lm_3.set_xlim(np.nanmin(model_fitted_y)-0.5, np.nanmax(model_fitted_y)+0.2)
	plot_lm_3.set_ylim(np.nanmin(model_norm_residuals_abs_sqrt)-0.1, np.nanmax(model_norm_residuals_abs_sqrt)+0.2)

	# annotations
	if np.isnan(model_norm_residuals_abs_sqrt).any():
	    model_norm_residuals_abs_sqrt[np.argwhere(np.isnan(model_norm_residuals_abs_sqrt))]=0


	abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
	abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
	for r, i in enumerate(abs_sq_norm_resid_top_3):
		plot_lm_3.annotate(indexes[i],xy=(model_fitted_y.iloc[i],model_norm_residuals_abs_sqrt[i]))



	############################
	#### Leverage plot
	############################


	#     shenanigans for cook's distance contours
	def graph(formula, x_range, label=None):
	    x = x_range
	    y = formula(x)
	    plot_lm_4.plot(x, y, label=label, lw=1, ls='--', color='red')



	# annotations

	plot_lm_4 = axes[1,1]


	plot_lm_4.scatter(x=model_leverage, y=model_norm_residuals, alpha=0.5)


	from statsmodels.nonparametric.smoothers_lowess import lowess

	w = lowess(endog=model_norm_residuals,exog=model_leverage,return_sorted=True,frac=2./3.,it=0)
	plot_lm_4.plot(w[:,0], w[:,1],color='red' )

	leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

	for i in leverage_top_3:
	    plot_lm_4.annotate(indexes[i],xy=(model_leverage[i],
	                                   model_norm_residuals[i]))


	p = len(results.params) # number of model parameters

	graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
	      np.linspace(np.min(model_leverage), np.max(model_leverage), 100)) # 0.5 line

	graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
	      np.linspace(np.min(model_leverage), np.max(model_leverage), 100)) # 1 line
	graph(lambda x: -np.sqrt((0.5 * p * (1 - x)) / x),
	      np.linspace(np.min(model_leverage), np.max(model_leverage), 100),
	      'Cook\'s distance') # 0.5 line
	graph(lambda x: -np.sqrt((1 * p * (1 - x)) / x),
	      np.linspace(np.min(model_leverage), np.max(model_leverage), 100)) # 1 line


	plot_lm_4.legend(loc='upper right')
	plot_lm_4.set_xlim(np.min(model_leverage)-0.0001, np.max(model_leverage)+0.0001)
	plot_lm_4.set_ylim(np.nanmin(model_norm_residuals)-0.005, np.nanmax(model_norm_residuals)+0.005)
	plot_lm_4.set_title('Residuals vs Leverage')
	plot_lm_4.set_xlabel('Leverage')
	plot_lm_4.set_ylabel('Standardized Residuals')

	plt.show()
