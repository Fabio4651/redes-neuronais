import pandas as pd
import numpy as np
from sklearn import preprocessing
from pylab import plot, legend, subplot, grid, xlabel, ylabel, show, title
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.nodes import BiasNode, Connection
import matplotlib.pyplot as plt
from pyneurgen.recurrent import NARXRecurrent
#%matplotlib inline


df = pd.read_csv("air/dataset.csv",index_col=False, parse_dates=[['day','month','year','hour']])
df.shape[:]

df['day_month_year_hour'] = pd.to_datetime(df.day_month_year_hour , format = '%d %m %Y %H')
data = df.drop(['day_month_year_hour'], axis=1)
data.index = df.day_month_year_hour
data = data.drop(['No'],axis=1)

data = data.fillna(method='bfill')

data = data.drop(['DEWP','TEMP','cbwd','Iws','Ir'],axis=1)
data.head(n=5)

# data['PRES']=pd.DataFrame(data['PRES'].values/np.max(data['PRES'].values))
# data['Is']=pd.DataFrame(data['Is'].values/np.max(data['Is'].values))
all_inputs = pd.DataFrame(data.values[:,1:])
all_targets = pd.DataFrame(data.values[:,0]) 



all_inputs[0] = all_inputs[0]/all_inputs[0].max()
all_inputs[1] = all_inputs[1]/all_inputs[1].max()
all_inputs.head(n=5)

all_targets[0] = all_targets[0]/all_targets[0].max()
all_targets.head(n=5)

print(all_inputs[0])

# print(all_inputs.shape[:])
# print(all_targets.shape[:])
# print(all_inputs.dtypes)

######################################################

# input_nodes = 10
# hidden_nodes = 4
# output_nodes = 1

# output_order = 3
# incoming_weight_from_output = .6
# input_order = 2
# incoming_weight_from_input = .4

# net = NeuralNet()
# net.init_layers(input_nodes, [hidden_nodes], output_nodes,
#         NARXRecurrent(
#             output_order,
#             incoming_weight_from_output,
#             input_order,
#             incoming_weight_from_input))

# net.randomize_network()

# net.set_all_inputs(all_inputs.values)
# net.set_all_targets(all_targets.values)

# net.set_learnrate(.1)

# length = len(all_inputs)
# learn_end_point = int(length * 0.8)

# net.set_learn_range(0, learn_end_point)
# net.set_test_range(learn_end_point + 1, length - 1)

# net.layers[1].set_activation_type('tanh')

# net.learn(epochs=10, show_epoch_results=True,random_testing=False)

# mse = net.test()

# np.sqrt(mse)

# test_positions = [item[0][1] * 1000.0 for item in net.get_test_data()]

# all_targets1 = [item[0][0] for item in net.test_targets_activations]
# allactuals = [item[1][0] for item in net.test_targets_activations]
# plt.figure(figsize=(9,7))
# subplot(2, 1, 1)
# plot(test_positions, all_targets1, 'bo', label='targets')
# plot(test_positions, allactuals, 'ro', label='actuals')
# grid(True)
# legend(loc='upper right', numpoints=1)
# title("Test Target Points vs Actual Points")
# show()

# plt.figure(figsize=(9,7))
# subplot(2, 1, 2)
# plot(range(1, len(net.accum_mse) + 1, 1), np.sqrt(net.accum_mse))
# xlabel('epochs')
# ylabel('Root mean squared error')
# grid(True)
# title("Root Mean Squared Error by Epoch")

# show()























