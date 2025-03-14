
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

learning_rate = 0.01
training_epochs = 50

trX = np.linspace(-3, 3, 101)

num_coefficients = 3
trY_coefficients = [1.5, 1, -0.5]    # [x**0, x**1, x**2]
trY = 0

for i in range(num_coefficients):
    trY += trY_coefficients[i] * np.power(trX, i)

trY += np.random.randn(*trX.shape) * 1.

tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)


def model(X, w):
    terms = []
    for i in range(num_coefficients):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


w = tf.Variable([0.] * num_coefficients)
y_model = model(X, w)

# cost = tf.square(tf.pow(Y - y_model, 2))
cost = tf.square(Y - y_model)
train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    for (х, у) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: х, Y: у})

w_val = sess.run(w)
print(w_val)

print("\nВеличина неточности")
print([float(trY_coefficients[i] - w_val[i]) for i in range(num_coefficients)])

sess.close()

# plt.scatter(trX, trY)
trY2 = 0
for i in range(num_coefficients):
    trY2 += w_val[i] * np.power(trX, i)
plt.plot(trX, trY2, color='r', linewidth=3)
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()
