#%%
from yaml import load, Loader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

#%%
with open('/home/mv23682/Documents/Abil_Calcif/configuration/ensemble_regressor.yml', 'r') as f:
    model_config = load(f, Loader=Loader)
    
run_name = 'wo_MLP/w_log'
species = '19_Mar_obs'
df = pd.read_csv(model_config['local_root'] + model_config['path_out'] + run_name + "/" + species + ".csv")

try:
    df = df.rename({'level_0': 'time','level_1':'depth','level_2':'lat','level_3':'lon'})
except:
    pass
predictors = model_config['predictors']
plt.rcParams.update({'font.size':20})

#%% Define scatter function
def loglog(df,x,y,
            ax=None, fig=None,
            title=None,axis1=None,axis2=None,
            xmin=None,xmax=None,ymin=None,ymax=None, single_plot=False):
    # Drop rows with missing values
    df = df.dropna(subset=[x, y])
    X = df[x]
    Y = df[y]

    # Calculate Spearman's rank correlation coefficient
    spearman_corr, _ = spearmanr(X, Y)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(X, Y))

    # Calculate R-squared
    mean_y = np.mean(Y)
    ss_tot = np.sum((Y - mean_y) ** 2)
    ss_res = np.sum((Y - X) ** 2)
    r_squared = 1 - (ss_res / ss_tot)   

    # Convert coefficients to strings
    spearman_corr_str = f'Spearman: {spearman_corr:.2f}'
    rmse_str = f'RMSE: {rmse:.2f}'
    r_squared_str = f'R$^2$: {r_squared:.2f}'
    
    if single_plot:
        if ax is None:
            fig, ax = plt.subplots()

        p = ax.loglog(X, Y, 'bo', markersize=1.5)
        
        # Add title
        if title is None:
            ax.set_title(x, fontdict={"fontsize": 18})
        else:
            ax.set_title(title, fontdict={"fontsize": 18})

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.loglog([xmin, xmax], [ymin, ymax], 'k--', lw=1)
        ax.set_xlabel('Obs')
        ax.set_ylabel('Model')

        # Add Spearman's rank correlation coefficient and RMSE to the top left corner
        ax.text(xmin + 0.02, ymax - 0.12*(ymax-ymin), f'{r_squared_str}\n{spearman_corr_str}\n{rmse_str}', fontsize=18, va='top', ha='left')


    else:
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        else:
            if hasattr(ax, '__iter__'):
                ax1, ax2 = ax
            else:
                ax1, ax2 = ax, None

        try:
            if ax2 is not None:
                p1 = ax1.loglog(X, Y, 'bo', markersize=1.5)
                p2 = ax2.loglog(X, Y, 'bo', markersize=1.5)
            else:
                p = ax1.loglog(X, Y, 'bo', markersize=1.5)

            # Add title for the specified subplot
            if title is None:
                ax1.set_title(x, fontdict={"fontsize": 18})
                if ax2 is not None:
                    ax2.set_title(y, fontdict={"fontsize": 18})
            else:
                ax1.set_title(title, fontdict={"fontsize": 18})
                if ax2 is not None:
                    ax2.set_title(title, fontdict={"fontsize": 18})

        except TypeError:  # if ax is not an array, it's a single axis
            p = ax1.loglog(X, Y, 'bo', markersize=1.5)

            # Add title for the specified subplot
            if title is None:
                ax1.set_title(x, fontdict={"fontsize": 24})
            else:
                ax1.set_title(title, fontdict={"fontsize": 24})

        # Set limits and labels for both subplots
        if ax1 is not None:
            ax1.set_xlim([xmin, xmax])
            ax1.set_ylim([ymin, ymax])
            ax1.loglog([xmin, xmax], [ymin, ymax], 'k--', lw=1)
            ax1.set_xlabel('Obs')
            ax1.set_ylabel('Model')

            # Add Spearman's rank correlation coefficient and RMSE to the top left corner
            ax1.text(xmin + 0.02, ymax - 0.12*(ymax-ymin), f'{r_squared_str}\n{spearman_corr_str}\n{rmse_str}', fontsize=18, va='top', ha='left')

        if ax2 is not None:
            ax2.set_xlim([xmin, xmax])
            ax2.set_ylim([ymin, ymax])
            ax2.loglog([xmin, xmax], [ymin, ymax], 'k--', lw=1)
            ax2.set_xlabel('Obs')
            ax2.set_ylabel('Model')
            # Add Spearman's rank correlation coefficient and RMSE to the top left corner
            ax2.text(xmin + 0.02, ymax - 0.12*(ymax-ymin), f'{r_squared_str}\n{spearman_corr_str}\n{rmse_str}', fontsize=18, va='top', ha='left')


    return fig, (ax1, ax2) if not single_plot else (ax,) # Return the figure and axes objects if needed


# %% Plot Model versus observations

fig, ax = plt.subplots(1,2)
fig.suptitle('Model vs. Observations')
fig.set_size_inches(15,7.5)
fig.tight_layout(h_pad=2)

loglog(df,'Calcification','Calcification_mod',
        ax=ax[0],
        fig=fig,
        title=None,
        axis1=0,
        xmin=0.1,
        xmax=2000,
        ymin=0.1,
        ymax=2000)


loglog(df,'Primary_Production','Primary_Production_mod',
        ax=ax[1],
        fig=None,
        title=None,
        axis1=1,
        xmin=4,
        xmax=10000,
        ymin=4,
        ymax=10000)

plt.show()
# %%
fig,ax = plt.subplots()
fig.set_size_inches(7.5,7.5)
loglog(df,'Calcification','Calcification_mod',
        ax=ax,
        fig=fig,
        title=r"Calcium Carbonate Production ($\mathrm{\mu}$mol C m$^{-3}$ d$^{-1}$)",
        axis1=0,
        xmin=0.1,
        xmax=2000,
        ymin=0.1,
        ymax=2000,
        single_plot=True)
plt.show()
# %%
