import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import axes3d

# Sliding Window Method

# second CJ file
df1 = pd.read_csv('20161207111012-59968-right-speed_0.500.csv')
df2 = pd.read_csv('20161207111143-59968-right-speed_0.500.csv')
df3 = pd.read_csv('20161207112226-59968-right-speed_0.500.csv')
df4 = pd.read_csv('20161207112544-59968-right-speed_0.500.csv')

df5 = pd.read_csv('20161207113602-59968-right-speed_0.500.csv')

# data importing and preprocessing
data_1 = df1.loc[:, ['elapsed_time', '8_x', '8_y', '8_z', '8_c']]
data_2 = df2.loc[:, ['elapsed_time', '8_x', '8_y', '8_z', '8_c']]
data_3 = df3.loc[:, ['elapsed_time', '8_x', '8_y', '8_z', '8_c']]
data_4 = df4.loc[:, ['elapsed_time', '8_x', '8_y', '8_z', '8_c']]

data_5 = df5.loc[:, ['elapsed_time', '8_x', '8_y', '8_z', '8_c']]

data_1 = data_1[data_1['8_c'] > 0]
data_2 = data_2[data_2['8_c'] > 0]
data_3 = data_3[data_3['8_c'] > 0]
data_4 = data_4[data_4['8_c'] > 0]

data_5 = data_5[data_5['8_c'] > 0]

data = pd.concat([data_1, data_2, data_3, data_4])
print("data.shape:", data.shape)

# print(data_1['8_x'])

X_dt = data['elapsed_time']
y_dt = data.loc[:, ['8_x', '8_y', '8_z']]
print(X_dt.shape, y_dt.shape)

print("X:", X_dt)
print("y:", y_dt)

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

# window setup
window = []
test = []
i = 0
while i + 1 <= 17.5:
    window.append(data.loc[(data['elapsed_time'] >= i) & (data['elapsed_time'] < i + 1)])
    test.append(data_5.loc[(data_5['elapsed_time'] >= i) & (data_5['elapsed_time'] < i + 1)])
    i += 0.5

print("Window length:", len(window))
print("test length:", len(test))

# print("window:\n", window[33])
print('window shape:', window[33].shape)

# print("test:\n", test[33])
print('test shape:', test[33].shape)

# kernel setup
gpr = []
kernel = 1.0 * RBF(length_scale=10, length_scale_bounds=(1e-1, 1e6))
for j in range(len(window)):
    gpr.append(GaussianProcessRegressor(kernel=kernel, alpha=1e-3, optimizer='fmin_l_bfgs_b',
                                        n_restarts_optimizer=20, random_state=0))

print("gpr:", gpr[1])

# fitting
k1 = []
k2 = []
for j in range(len(window)):
    train = window[j]
    X = np.array(train['elapsed_time']).reshape(-1, 1)
    y = np.array(train.loc[:, ['8_x', '8_y', '8_z']])
    gpr[j].fit(X, y)
    print("kernel", j, "params:", gpr[j].kernel_.get_params())
    k1.append(gpr[j].kernel_.get_params()['k1__constant_value'])
    k2.append(gpr[j].kernel_.get_params()['k2__length_scale'])

# predicting
X_inputs = []
y_predictions = []
y_covs = []
outputs = []
covs = []
for j in range(len(window)):
    pred = test[j]
    X_test = np.array(pred['elapsed_time'])
    X_inputs.append(X_test)

    y_test = np.array(pred.loc[:, ['8_x', '8_y', '8_z']])
    y_pred, cov = gpr[j].predict(X_test.reshape(-1, 1), return_cov=True)

    y_predictions.append(y_pred)
    y_covs.append(cov)

    outputs.append(np.insert(y_pred, 0, X_test, axis=1))
    covs.append(np.insert(np.diag(cov).reshape(-1, 1), 0, X_test, axis=1))

print("outputs[1] shape:", outputs[1].shape)

# recombining into different chunks
df_chunk = []
cov_chunk = []
for j in range(len(window)):
    df_chunk.append(pd.DataFrame(outputs[j], columns=['elapsed_time', '8_x', '8_y', '8_z']))
    cov_chunk.append(pd.DataFrame(covs[j], columns=['elapsed_time', 'covariance']))

print("df_chunk[0]:", df_chunk[0])
print("df_chunk[1]:", df_chunk[1])

# create two dictionaries to average the predicted mean and variance inside the overlapping windows:
# the keys are the time t we want to look at; for the first dictionary, the values are the
# corresponding 3d coordinates, which are set to zeros initially; for the second dictionary,
# the values are the corresponding variance, which are also set to zeros initially.
d = dict(zip(np.array(data_5['elapsed_time']), np.zeros((data_5.shape[0], 3))))
var = dict(zip(np.array(data_5['elapsed_time']), np.zeros((data_5.shape[0], 1))))
# print('d:', d)
# print('var:', var)

# reading values inside the chunks into dictionaries, keep the means and variances
# for time below 0.5 or greater than 17.0 (including 17.0), and average the means
# and variances in between
for t in d:
    if t < 0.5:
        chunk = df_chunk[0]
        variance = cov_chunk[0]
        d[t] = np.array(chunk[chunk['elapsed_time'] == t].iloc[:, [1, 2, 3]])
        var[t] = np.array(variance[variance['elapsed_time'] == t].iloc[:, 1])
    elif t >= 17.0:
        chunk = df_chunk[33]
        variance = cov_chunk[33]
        d[t] = np.array(chunk[chunk['elapsed_time'] == t].iloc[:, [1, 2, 3]])
        var[t] = np.array(variance[variance['elapsed_time'] == t].iloc[:, 1])
    else:
        index = int(t // 0.5)
        chunk1 = df_chunk[index - 1]
        chunk2 = df_chunk[index]
        variance1 = cov_chunk[index - 1]
        variance2 = cov_chunk[index]

        arr1 = np.array(chunk1[chunk1['elapsed_time'] == t].iloc[:, [1, 2, 3]])
        arr2 = np.array(chunk2[chunk2['elapsed_time'] == t].iloc[:, [1, 2, 3]])
        avg_arr = np.mean(np.concatenate((arr1, arr2)), axis=0)
        d[t] = np.atleast_2d(avg_arr)

        v1 = np.array(variance1[variance1['elapsed_time'] == t].iloc[:, 1])
        v2 = np.array(variance2[variance2['elapsed_time'] == t].iloc[:, 1])
        v_arr = np.mean(np.concatenate((v1, v2)), axis=0)
        var[t] = np.atleast_1d(v_arr)

# print('updated d:', d)
# print('updated var:', var)

# append the results got from the dictionaries above into different lists
X_results = []
y_results = []
var_results = []
for t in d:
    X_results.append(t)
    y_results.append(d[t][0])
    var_results.append(var[t])

# transform the lists into numpy arrays
X_results = np.array(X_results)
y_results = np.array(y_results)
var_results = np.array(var_results)
# print('X_results:', X_results)
# print('y_results:', y_results)
print('y_results shape:', y_results.shape)
# print('var_results:', var_results)
print('var_results shape:', var_results.shape)

# compute the mean squared errors in each axis and in total
y_true = np.array(data_5.loc[:, ['8_x', '8_y', '8_z']])
print('MSE in x:', mean_squared_error(y_results[:, 0], y_true[:, 0]))
print('MSE in y:', mean_squared_error(y_results[:, 1], y_true[:, 1]))
print('MSE in z:', mean_squared_error(y_results[:, 2], y_true[:, 2]))
print('MSE total:', mean_squared_error(y_results, y_true) * 3)

# compute correlations
print('Correlation in x:\n', np.corrcoef(y_results[:, 0], y_true[:, 0]))
print('Correlation in y:\n', np.corrcoef(y_results[:, 1], y_true[:, 1]))
print('Correlation in z:\n', np.corrcoef(y_results[:, 2], y_true[:, 2]))

# plot results in 2d
plt.figure()
plt.plot(X_results, y_results[:,0], c='blue', label='x curve prediction')
plt.plot(X_results, y_results[:,1], c='orange', label='y curve prediction')
plt.plot(X_results, y_results[:,2], c='green', label='z curve prediction')
plt.plot(X_results, y_true[:,0], c='tomato', label='true x curve')
plt.plot(X_results, y_true[:,1], c='orchid', label='true y curve')
plt.plot(X_results, y_true[:,2], c='peru', label='true z curve')
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.show()

# plot variance over time
plt.figure()
plt.plot(X_results, var_results, label='variances')
plt.legend()
plt.show()

# plot results in 3d
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data_5['8_x'], data_5['8_y'], data_5['8_z'], s=0.5, label='actual values')
ax.scatter3D(pd.Series(y_results[:, 0]), pd.Series(y_results[:, 1]), pd.Series(y_results[:, 2]), s=0.5,
             label='predictions')
ax.set_xlabel('8_x')
ax.set_ylabel('8_y')
ax.set_zlabel('8_z')
plt.legend()
plt.show()

# plot kernel parameters
plt.figure()
plt.plot(np.arange(0,34,1), np.array(k1), c='gold', label='sigma^2')
plt.ylim(0,2)
plt.legend()

plt.figure()
plt.plot(np.arange(0,34,1), np.array(k2), c='orange', label='length scale')
plt.legend()
plt.show()

