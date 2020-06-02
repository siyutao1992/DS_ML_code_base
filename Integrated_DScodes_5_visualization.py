###############################################################################################
#### Visualization with matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.subplot(2, 1, 1) # nrows, ncols, nsubplot
plt.plot(x,y) # line plot
plt.subplot(2, 1, 2) # nrows, ncols, nsubplot
plt.scatter(x,y) # scatter plot
plt.hist(values, bins = 3) # histogram plot
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.yticks([0, 2, 4, 6, 8, 10], \
    ['0', '2B', '4B', '6B', '8B', '10B'])
plt.xlim((1947, 1957))
plt.ylim((0, 1000))
plt.axis((1947, 1957, 0, 600))
plt.axis('equal')
plt.legend(loc='upper right')
plt.annotate('setosa', xy=(5.0, 3.5)) # put text on the figure
plt.annotate('setosa', xy=(5.0, 3.5), xytext=(4.25, 4.0), arrowprops={'color':'red'}) # use arrow with text
    # xy is the end point of arrow, xytext is the position of the text (start point of arrow)
plt.show()

plt.plot(pandas_series) # can directly plot pandas Series. The indexes would be used as x-axis data
df.plot() # plot all columns in df at once. Different lines would be indicated with lengend
df.plot(subplots=True) # plot different columns in different subplots
df.plot(x='col_name1', y='col_name2', kind='scatter')
df.plot(y='col_name2', kind='box')
df.plot(y='col_name2', kind='hist', bins=30, range=(4,8), cumulative=True, normed=True)
df['col1'].plot(color='r', style=‘.’, legend=True, label='setosa')
df['col2'].plot(color='b', style=‘.-’, legend=True, label='virginica') # plots would be in the same figure
plt.xlabel('my_xlabel')
plt.ylabel('my_ylabel')
plt.yscale('log')
plt.savefig('df_plot.png') # png or jpg or pdf
plt.show()

## plot styles
# color: b, g, r, c (cyan)
# marker: o (circle), * (star), s (square), + (plus)
# line: : (dotted), - (dashed)

###############################################################################################
#### visualization with seaborn
import seaborn as sns
import numpy as np
sns.set()
tips =sns.load_dataset('tips') # load data frame
sns.lmplot(x='total_bill', y='tip', data=tips) # show confidence interval
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='Set1') # group factor by column 'sex'
sns.lmplot(x='total_bill', y='tip', data=tips, col='sex') # separate plots for different 'sex' levels
sns.residplot(x='age', y='fare', data=tips) # plot residual of linear regression
# below are univariate plots (for distribution visualization)
np.percentile(df['col_name'], [25, 50, 75]) # print percentiles
sns.boxplot(x='qual_col', y='quant_col', data=df)
sns.stripplot(x='day', y='tip', data=tips, size=4, jitter=True) # plot 'tip' distributions categorized by 'day'
sns.swarmplot(x='day', y='tip', data=tips, hue='sex', orient='h') # grouped by different 'sex' levels
sns.violinplot(x='day', y='tip', data=tips, inner=None, color='lightgray')
# below are multivariate plots (for visualizing joint distributions)
sns.jointplot(x= 'total_bill', y= 'tip', data=tips)
sns.pairplot(tips, hue='sex') # only show numerical variables
np.corrcoef(data_mat) # calc corr mat to show correlation between vars








