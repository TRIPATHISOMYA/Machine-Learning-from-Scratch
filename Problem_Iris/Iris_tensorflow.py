import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import model_selection

np.random.seed(1100)
tf.set_random_seed(1100)
interval = 45
epoch = 450


# Loading the iris_data
iris_data = pandas.read_csv('../data/iris_data.csv')
iris_data = pd.get_dummies(iris_data, columns=['Species']) # One Hot Encoding
values = [iris_data.columns.values]


data = iris_data.values
X = data[:,0:4]
Y = data[:,4]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)



sess = tf.Session()

X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)


hidden_layer_nodes = 8


w1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes])) 
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,3])) 
b2 = tf.Variable(tf.random_normal(shape=[3]))  


hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))


loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


init = tf.global_variables_initializer()
sess.run(init)

# Training
print('Training the model...')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if i % interval == 0:
        print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))

# Prediction
print()
for i in range(len(X_test)):
print('Actual:', y_test[i], 'Predicted:', np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]})))
