# AI_3_Report

## 定义模型    **Keras的Sequential模型**

序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”。

可以通过向Sequential模型传递一个layer的list来构造该模型

也可以通过.add()方法一个个的将layer加入模型中

模型需要知道输入数据的shape，因此，Sequential的第一层需要接受一个关于输入数据shape的参数.

传递一个input_shape的关键字参数给第一层

有些2D层，如Dense，支持通过指定其输入维度input_dim来隐含的指定输入数据shape。

如果你需要为输入指定一个固定大小的batch_size（常用于stateful RNN网络），可以传递batch_size参数到一个层中，例如你想指定输入张量的batch大小是32，数据shape是（6，8），则你需要传递batch_size=32和input_shape=(6,8)

![Deep Learning](pictures/dp)

# 1. 基于RNN实现文本分类任务

Datasets：搜狐新闻数据  datasets/sohu.csv

Model： Keras的Sequential模型

Procedure： 

- 数据读取
```python
data = pd.read_csv('sohu.csv')
```
- 数据预处理
```python
#分词
for i in data['text']:
    i = jieba.cut(i)
data['text'] = data['text'].apply(lambda x:' '.join(jieba.cut(x)))
max_features = 10000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['text'])
list_tokenized_train = tokenizer.texts_to_sequences(data['text'])

from keras.preprocessing.sequence import pad_sequences
len = 100
x = pad_sequences(list_tokenized_train, maxlen = len)
y = data['label']
```

- 分离训练集与测试集
```python
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.3)

y1_train_onehot = np_utils.to_categorical(y1_train)   #转为二元矩阵
y1_test_onehot = np_utils.to_categorical(y1_test)
```
- 训练模型
```python
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(3,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
- 评估模型
```python
accuracy = model.evaluate(x1_test, y1_test_onehot, batch_size = 50)
print("test accuracy:{}".format(accuracy[1]))
>>>1271/1271 [==============================] - 0s 305us/step
test accuracy:0.9766588117267839
```

# 2. 基于CIFAR-10数据集使用CNN完成图像分类任务

Datasets: datasets/cifar-10-batches-py

Model: Keras的Sequential模型

Procedure: 

 - 1. 数据预处理
```python
def unpickle(file):
      with open(file, 'rb') as fo:
              dict = pickle.load(fo, encoding='bytes')
      return dict
 ```
 ```python
# 测试集文件
test_batch = unpickle(R'C:\Users\CZY\Desktop\AI-3\cifar-10-batches-py/test_batch')

X_test = np.array(test_batch[b'data'])
y_test = np.array(test_batch[b'labels']) # list type
```
```python
# 训练集文件
train_files = ['data_batch_'+str(i) for i in range(1,6)]

train_data_list,train_label_list = [],[]

for f in train_files:
    
    fpath = r'C:\Users\CZY\Desktop\AI-3\cifar-10-batches-py/' + f
    batch_dict = unpickle(fpath)
    
    batch_data = batch_dict[b'data']
    batch_labels = batch_dict[b'labels']
    train_data_list.append(batch_data)
    train_label_list.append(batch_labels)

X_train = np.concatenate(train_data_list, axis = 0)
y_train = np.concatenate(train_label_list, axis = 0)
```

- 2. 构建模型
```python
#模型的构建与编译
def base_model(opt):
    model = Sequential()
    
    # 32个卷积核(feature maps),步长为1，特征图的大小不会改变（周边补充空白），
    model.add(Conv2D(32,(3,3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))

    # channel是在前面 (theano后台)
    MaxPooling2D(pool_size=(2, 2), data_format="channels_first")
    model.add(Dropout(0.25))
    
    # 64个卷积核
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation('relu'))
    
    MaxPooling2D(pool_size=(2, 2), data_format="channels_first")
    model.add(Dropout(0.25))
    
    model.add(Flatten())   # Flatten layer
    model.add(Dropout(0.25))
    model.add(Dense(512))  # fully connected layer with 512 units
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']) # 要优化的是准确率
    return model
```

```python
# 初始化 RMSprop 优化器
opt1 = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# 初始化 Adam 优化器
# opt2 = keras.optimizers.Adam(lr=0.0001)
# 用RMSProp训练模型
cnn2 = base_model(opt1)
cnn2.summary() # 打印网络结构及其内部参数
# 进行100轮批次为32的训练,默认训练过程中会使用正则化防止过拟合            
history = cnn2.fit(X_train[:1000], y_train[:1000], 
                    epochs = 100, batch_size = 32, 
                    validation_data=(X_test[:200],y_test[:200]), 
                    shuffle=True)
```

```python
score2 = cnn2.evaluate(X_test,y_test)
print("损失值为{0:.2f},准确率为{1:.2%}".format(score2[0],score2[1]))
>>>
```

# 3. 基于MNIST数据集使用GAN实现手写图像生成的任务

Datasets: datasets/mnist

Model: Keras的Sequential模型

Procedure: 

-  数据预处理
```python
#读取数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Users\CZY\Desktop\AI_project3\MNIST_data/')
```
- 生成器和判别器

```python
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    """
    生成器
    
    noise_img: 生成器的输入
    n_units: 隐层单元个数
    out_dim: 生成器输出tensor的size，这里应该为32*32=784
    alpha: leaky ReLU系数
    """
    with tf.variable_scope("generator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(noise_img, n_units)
        # leaky ReLU
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        # dropout
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        # logits & outputs
        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)
        
        return logits, outputs
```

```python
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    """
    判别器
    
    n_units: 隐层结点数量
    alpha: Leaky ReLU系数
    """
    
    with tf.variable_scope("discriminator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        
        # logits & outputs
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)
        
        return logits, outputs
```
- 训练

```python
# batch_size
batch_size = 64
# 训练迭代轮数
epochs = 300
# 抽取样本数
n_sample = 25

# 存储测试样例
samples = []
# 存储loss
losses = []
# 保存生成器变量
saver = tf.train.Saver(var_list = g_vars)
# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            batch_images = batch[0].reshape((batch_size, 784))
            # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
            batch_images = batch_images*2 - 1
            
            # generator的输入噪声
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
            
            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})
        
        # 每一轮结束计算loss
        train_loss_d = sess.run(d_loss, 
                                feed_dict = {real_img: batch_images, 
                                             noise_img: batch_noise})
        # real img loss
        train_loss_d_real = sess.run(d_loss_real, 
                                     feed_dict = {real_img: batch_images, 
                                                 noise_img: batch_noise})
        # fake img loss
        train_loss_d_fake = sess.run(d_loss_fake, 
                                    feed_dict = {real_img: batch_images, 
                                                 noise_img: batch_noise})
        # generator loss
        train_loss_g = sess.run(g_loss, 
                                feed_dict = {noise_img: batch_noise})
        
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
              "Generator Loss: {:.4f}".format(train_loss_g))    
        # 记录各类loss值
        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
        
        # 抽取样本后期进行观察
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                               feed_dict={noise_img: sample_noise})
        samples.append(gen_samples)
        
        # 存储checkpoints
        saver.save(sess, './checkpoints/generator.ckpt')

# 将sample的生成数据记录下来
with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
```

-  损失函数变化
```python
fig, ax = plt.subplots(figsize=(20,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator Total Loss')
plt.plot(losses.T[1], label='Discriminator Real Loss')
plt.plot(losses.T[2], label='Discriminator Fake Loss')
plt.plot(losses.T[3], label='Generator')
plt.title("Training Losses")
plt.legend()
```
![ax](pictures/ax)
- 结果展示
```python
def view_samples(epoch, samples):
    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]): # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    return fig, axes
    
  # 加载生成器变量
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
    gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                           feed_dict={noise_img: sample_noise})
     
_ = view_samples(-1, samples) # 显示最后一轮的outputs
```

![results](res)
