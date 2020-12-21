import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import axes3d

# Global Window Method

# second CJ file
df1 = pd.read_csv('20161207111012-59968-right-speed_0.500.csv')
df2 = pd.read_csv('20161207111143-59968-right-speed_0.500.csv')
df3 = pd.read_csv('20161207112226-59968-right-speed_0.500.csv')
df4 = pd.read_csv('20161207112544-59968-right-speed_0.500.csv')

df5 = pd.read_csv('20161207113602-59968-right-speed_0.500.csv')

# data importing and preprocessing
data_1 = df1.loc[:,['elapsed_time','8_x','8_y','8_z','8_c']]
data_2 = df2.loc[:,['elapsed_time','8_x','8_y','8_z','8_c']]
data_3 = df3.loc[:,['elapsed_time','8_x','8_y','8_z','8_c']]
data_4 = df4.loc[:,['elapsed_time','8_x','8_y','8_z','8_c']]

data_5 = df5.loc[:,['elapsed_time','8_x','8_y','8_z','8_c']]

data_1 = data_1[data_1['8_c'] > 0]
data_2 = data_2[data_2['8_c'] > 0]
data_3 = data_3[data_3['8_c'] > 0]
data_4 = data_4[data_4['8_c'] > 0]

data_5 = data_5[data_5['8_c'] > 0]

data = pd.concat([data_1, data_2, data_3, data_4])
data = data.sample(frac=0.50, random_state=0)
print(data.shape)

# print(data_1['8_x'])

X_dt = np.array(data['elapsed_time']).reshape(-1,1)
y_dt = data.loc[:, ['8_x','8_y','8_z']]
print(X_dt.shape, y_dt.shape)
print(y_dt['8_x'])

# plot figure in 3d
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data_1['8_x'], data_1['8_y'], data_1['8_z'], s=0.5, label='data_1')
ax.scatter3D(data_2['8_x'], data_2['8_y'], data_2['8_z'], s=0.5, label='data_2')
ax.scatter3D(data_3['8_x'], data_3['8_y'], data_3['8_z'], s=0.5, label='data_3')
ax.scatter3D(data_4['8_x'], data_4['8_y'], data_4['8_z'], s=0.5, label='data_4')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()

# plot figure in 2d
plt.figure()
plt.scatter(X_dt, y_dt['8_x'], s=0.5, label='x in training trials')
plt.scatter(X_dt, y_dt['8_y'], s=0.5, label='y in training trials')
plt.scatter(X_dt, y_dt['8_z'], s=0.5, label='z in training trials')
plt.legend(loc='lower right')
plt.show()

# set up kernel, fit data and make predictions
kernel = 1.0 * RBF(length_scale=10, length_scale_bounds=(1e-1, 1e6))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, optimizer='fmin_l_bfgs_b',
                               n_restarts_optimizer=20, random_state=0)
gpr.fit(X_dt, y_dt)

data_5 = data_5[data_5['8_c'] > 0]
X_pred = np.array(data_5['elapsed_time']).reshape(-1,1)
y_pred, cov = gpr.predict(X_pred, return_cov=True)
variances = np.diag(cov)

# print(gpr.get_params())
print("kernel params:", gpr.kernel_.get_params())
print("cov shape:", cov.shape)

# print kernel parameters
print("sigma^2:", gpr.kernel_.get_params()['k1__constant_value'])
print("length scale:", gpr.kernel_.get_params()['k2__length_scale'])
####
# compute mean squares errors
y_true = np.array(data_5.loc[:, ['8_x','8_y','8_z']])
print('MSE in x:', mean_squared_error(y_pred[:,0], y_true[:,0]))
print('MSE in y:', mean_squared_error(y_pred[:,1], y_true[:,1]))
print('MSE in z:', mean_squared_error(y_pred[:,2], y_true[:,2]))
print('MSE total:', mean_squared_error(y_pred, y_true)*3)

# compute correlations
print('correlation in x:\n', np.corrcoef(y_pred[:,0], y_true[:,0]))
print('correlation in y:\n', np.corrcoef(y_pred[:,1], y_true[:,1]))
print('correlation in z:\n', np.corrcoef(y_pred[:,2], y_true[:,2]))

# plot results in 2d
plt.figure()
plt.plot(X_pred, y_pred[:,0], c='blue', label='x curve prediction')
plt.plot(X_pred, y_pred[:,1], c='orange', label='y curve prediction')
plt.plot(X_pred, y_pred[:,2], c='green', label='z curve prediction')
plt.plot(X_pred, y_true[:,0], c='tomato', label='true x curve')
plt.plot(X_pred, y_true[:,1], c='orchid', label='true y curve')
plt.plot(X_pred, y_true[:,2], c='peru', label='true z curve')
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.show()

# plot variance over time
plt.figure()
plt.plot(X_pred, variances, label='variances')
plt.legend()
plt.show()

# plot results in 3d
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data_5['8_x'], data_5['8_y'], data_5['8_z'], s=0.5, label='actual values')
ax.scatter3D(pd.Series(y_pred[:, 0]), pd.Series(y_pred[:, 1]), pd.Series(y_pred[:, 2]), s=0.5,
             label='predictions')
ax.set_xlabel('8_x')
ax.set_ylabel('8_y')
ax.set_zlabel('8_z')
plt.legend()
plt.show()

