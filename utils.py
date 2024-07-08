from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import medfilt
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_roc(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred) 
    roc_auc = auc(fpr, tpr)
    plt.figure()  
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Chance Prediction')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Precipitation Prediction')
    plt.legend()
    plt.show()

def transform_target(df, target, epsilon=1e-6):
    transforms = {f"sqrt_{target}": lambda x: np.sqrt(x), 
                  f"{target}^2": lambda x: np.power(x,2.0),
                  f"exp_{target}": lambda x: np.exp(x),
                  f"inv_{target}": lambda x: np.divide(1,np.add(x,epsilon)),
                  f"log_{target}": lambda x: np.log(np.add(x,epsilon))}
    for col, func in transforms.items():
        df.loc[:,col] = func(df[target].to_numpy())
    targets = list(transforms.keys()) + [target]
    return df, targets

def plot_feature_importance(names, values, title):
    y = [left for left, _ in sorted(zip(names,values), key=lambda x: x[1])]
    x = [right for _, right in sorted(zip(names,values), key=lambda x: x[1])]
    plt.barh(y=y, width=x)
    plt.grid()
    plt.title(title)
    plt.ylabel("Features")
    plt.xlabel("Importance")
    plt.show()

def transform_features(df, transforms):
    for col, func in transforms.items():
        df.loc[:,col] = func(df[col].to_numpy())
    return df

def hist_prob_plots(df, col):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    plt.subplot(1, 2, 1)
    plt.title(col)
    df[col].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.show()

def scatter_hist_plot(df, col, target):
    # function to plot a histogram and a scatter plot
    # side by side, for a certain variable
    plt.subplot(1, 2, 1)
    plt.title(col)
    df[col].hist()
    plt.subplot(1, 2, 2)
    plt.title(f"Corr to {target} = {df.corr().loc[col,target]}")
    plt.scatter(df[col], df[target])
    plt.show()

def hist_matrix(df, cols, nrow, ncol):
    for i, c in enumerate(cols):
        plt.subplot(nrow, ncol, i+1)
        plt.title(c, fontsize=10)
        df[c].hist()
    plt.tight_layout()
    plt.show()

def find_best_kernel(df, col, ref):
    kernel_sizes = [0,7,11,21,31,51,71,101,201,301,501,701,901,1001,3001]
    res = []
    for k in kernel_sizes:
        data = df.copy()
        if k > 0:
            data.loc[:,col] = medfilt(data[col].to_numpy(), kernel_size=k)
        corr = abs(data.corr()[ref][col])
        res.append((k,corr))
    res = sorted(res, key=lambda x: -x[1])
    print(f"{col} best kernel: {res[0]}")
    return res[0][0]

def derivative_computer(df, col, dt, dx_colname="dx"):
    DEFAULT = 2
    if dt < 1 or dt >= df.shape[0]:
        print("dt must be non-zero positive integer < rows in df")
        print(f"dt = {dt} is not supported, defaulting to {DEFAULT}")
        dt = DEFAULT
    dx_list = []
    for _, data in df.groupby(["unit"]):
        x = data.loc[:,col].to_numpy().reshape((data.shape[0],))
        dx = np.zeros(shape=(data.shape[0],))
        if dt < x.shape[0]:
            for i in range(dt, x.shape[0]):
                dx[i] = (x[i] - x[i-dt]) / dt
        dx_list.append(dx)
    
    df[dx_colname] = np.concatenate(dx_list)
    return df

def find_best_dt(df, col, target):
    dts = np.linspace(start=2, stop=200, num=100)
    res = []
    for dt in dts:
        dt = int(dt)
        data = df.loc[:,["unit",col,target]]
        dx_colname = col+"_dx"+str(dt) 
        data = derivative_computer(data, col, dt, dx_colname)
        corr = abs(data.corr()[target][dx_colname])
        res.append((dt,corr))
        del data
    res = sorted(res, key=lambda x: -x[1])
    print(f"{col} best dt: {res[0]}")
    return res[0][0]