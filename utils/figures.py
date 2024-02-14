import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import plotly.graph_objects as go
from scipy.stats import bootstrap
import numpy as np

def plot_predictions(y_true, y_pred, specimen, ci=True, scaler=None):
    '''Plot the ground truth vs the mean prediction with 95% CI
    
    Parameters:
    - y_true (np.ndarray): of shape (n,), where n is the number of time steps
    - y_pred (list): of length m=15 where each element is an np.ndarray of shape (n,)
    - specimen (str): name of the specimen
    - ci (bool): whether to plot the 95% CI

    Returns:
    - fig (matplotlib.figure.Figure): the figure object
    '''
    if scaler:
        # y_true = scaler.inverse_transform(y_true.reshape(-1,1)).reshape(-1,)
        y_pred = [scaler.inverse_transform(y.reshape(-1,1)).reshape(-1,) for y in y_pred]

    fig, ax = plt.subplots()
    ax.plot(y_true[::-1], y_true, label='Ground Truth', color='black')
    ax.plot(y_true[::-1], np.mean(y_pred, axis=0), label='Weighted Mean Estimate', color='#799ed3')
    
    if ci:
        y_pred = np.array(y_pred).reshape(1,15,-1)
        lower, upper = bootstrap(y_pred, np.mean, n_resamples=1000, method='percentile').confidence_interval
        ax.fill_between(y_true[::-1], lower, upper, alpha=0.3, color='#799ed3', label='95% CI')

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.set_title(f'Remaining Useful Life prediction for {specimen}')
    ax.set_xlabel('Cycles')
    ax.set_ylabel('RUL (cycles remaining)')
    ax.legend()
    return fig, lower, upper

def plot_hi_ga(X, y, **kwargs):
    '''Plot the hi_ga series for each specimen
    
    Parameters:
    - X (dict): dictionary where the keys are the specimen names and the values are the corresponding hi_ga sequences
    - y (dict): dictionary where the keys are the specimen names and the values are the corresponding RULs (reversed cycles)
    - **kwargs: optional arguments
        - linestyle (dict): dictionary where the keys are the specimen names ('ca', 'va', 'sp') and the values are the line styles
        - ncols (int): number of columns for the legend
        - legend_loc (str): location of the legend
        - xlabel (str): x-axis label
        - ylabel (str): y-axis label
    
    Returns:
    - fig (matplotlib.figure.Figure): the figure object
    - ax (matplotlib.axes._subplots.AxesSubplot): the axes object
    '''
    linestyle = kwargs.get('linestyle',None)
    ncols = kwargs.get('ncols',1)
    legend_loc = kwargs.get('legend_loc','lower right')
    xlabel = kwargs.get('xlabel','Fatigue Cycles')
    ylabel = kwargs.get('ylabel',r'HI$_{ga}$')

    fig, ax = plt.subplots()
    for specimen, hi_ga in X.items():
        cycles = y[specimen][::-1]
        line_style = linestyle[specimen.split('_')[0]] if linestyle else None
        ax.plot(cycles, hi_ga, label=specimen, linestyle=line_style)
    ax.legend(ncol=ncols, loc=legend_loc)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def plot_metric(metric, models, specimens):
    '''Compare the metric values of different models/ approaches for each test specimen

    Parameters:
    - metric (class): the metric class object, with the following attributes:
        - name (str): the name of the metric
        - value (dict): keys are the model names and values are the metric values of shape (n,) where n is the number of test specimens
    - models (list): of length m, where each element is the name of a model ('svr', 'lstmn')
    - specimens (list): of length n, where each element is the name of a test specimen

    Returns:
    - fig (plotly.graph_objects.Figure): the figure object
    '''
    metric_args = {
        'RMSE': {'title': 'Root Mean Squared Error'},
        'MAE': {'title': 'Mean Absolute Error'},
        'MAPE': {'title': 'Mean Absolute Percentage Error', 'yaxis': '.0%', 'hovertemp': '%{y:.2%}'},
        'CRA': {'title': 'Cumulative Relative Accuracy', 'yaxis': '.1f', 'hovertemp': '%{y:.2f}'},
        'CIDC': {'title': 'Confidence Interval Distance Convergence'},
        'PIB': {'title': 'Percentage of True Values within the Confidence Interval', 'yaxis': '.0%', 'hovertemp': '%{y:.2%}'},
        }
    
    colors = ['#808183', '#799ed3', '#46484b','#bcbdbf']
    
    fig = go.Figure()
    for model in models:
        fig.add_trace(go.Bar(x=specimens, y=metric.value[model], name=model, marker_color=colors.pop(0), hovertemplate=metric_args.get(metric.name, {}).get('hovertemp', '%{y:,.0f}')))
    # fig.add_trace(go.Bar(x=specimens, y=metric.value[models], name=models, marker_color='#808183', hovertemplate=metric_args.get(metric.name, {}).get('hovertemp', '%{y:,.0f}')))
    # fig.add_trace(go.Bar(x=specimens, y=metric.value[models], name=models, marker_color='#799ed3', hovertemplate=metric_args.get(metric.name, {}).get('hovertemp', '%{y:,.0f}')))
    fig.update_layout(barmode='group', title=metric_args.get(metric.name, {}).get('title', 'Metric'), xaxis_title='Specimen', yaxis_title=metric.name)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis=dict(tickformat=metric_args.get(metric.name, {}).get('yaxis', '')))
    return fig