部分作业的关键代码的参考答案在assignment2的cs231n/layers.py里面

assignment1主要是图片分类的几个算法：

一、k-Nearest Neighbor (kNN)

kNN算法很简单，就两步：

1.记忆训练集数据；

2.比较测试集数据和训练集数据寻找离测试集数据最近的k个训练集数据来判断测试集数据所属分类；

3.k是超参数，可以通过在验证集上cross-validation得出。

开头几块代码都是在导入包、导入数据、展示数据、预处理数据（将数据由3维图像拉成1维向量）。

之后是训练模型（一瞬间的事）。

接下来是测试模型。在不能使用np.linalg.norm()的情况下，要计算一组测试数据和一组训练集数据的距离，并且从其中找出对于每个测试数据最小的k个距离（排序取k个）。

作业中要求我们分别用两个循环、一个循环、不用循环三种方式计算距离。在k_nearest_neighbor.py文件中补充完整这三个函数。

答案：

两个循环：很简单的np向量运算

    dists[i][j] = np.sqrt(np.sum(np.square(X[i,:]-self.X_train[j,:])))

一个循环：需要注意因为扩大了一个维度，求和的时候要保留第一个维度不能求和，而是在第二个维度（axis=1）求和

    dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))

不用循环：为了计算方便，把(x-y)^2拆分为x^2+y^2-2xy。

    t_2xy=np.matmul(X, self.X_train.T)*(-2)
    t_x2=np.sum(np.square(X), axis=1, keepdims=True)
    t_y2=np.sum(np.square(self.X_train), axis=1)
    dists=t_2xy+t_x2+t_y2
    dists=np.sqrt(dists)

之后还需要把k_nearest_neighbor.py文件中的predict_labels函数补充完整：

答案：

前一个：

    closest_y = (self.y_train[np.argsort(dists[i])[:k]])

后一个：

    y_pred[i]=np.argmax(np.bincount(closest_y))

之后，要用Cross-validation求k的最佳值，补充完整代码：

前一空：

    X_train_folds = np.array_split(X_train,num_folds)
    y_train_folds = np.array_split(y_train,num_folds)

后一空：

    classifier = KNearestNeighbor()
    for k in k_choices:
        accuracy = []
        for fold in range(num_folds):
            t_X_train = X_train_folds[:]
            t_y_train = y_train_folds[:]
            X_val = t_X_train.pop(fold)
            y_val = t_y_train.pop(fold)
            t_X_train = np.array([a for b in t_X_train for a in b])
            t_y_train = np.array([a for b in t_y_train for a in b])
            classifier.train(t_X_train, t_y_train)
            y_val_pred = classifier.predict(X_val, k=k)
            num_correct = np.sum(y_val_pred == y_val)
            accuracy.append(float(num_correct) / y_val.shape[0])
        k_to_accuracies[k]=accuracy

最后，要选择表现最好的k（平均准确率最高的k）用在测试集上。

    best_k = 10

得到的测试集准确率：accuracy: 0.282000

p.s.还有几个Inline Question

**Inline Question 1** 

Notice the structured patterns in the distance matrix, where some rows or columns are visible brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)

- What in the data is the cause behind the distinctly bright rows?
- What causes the columns?

$\color{blue}{\textit Your Answer:}$ The cause is what discerns the differences between test data and training data, which is graphic difference. The columns means that there are a number of test graphs which have similar differences from train data, so there may be some major differences.

**Inline Question 2**

We can also use other distance metrics such as L1 distance.
For pixel values $p_{ij}^{(k)}$ at location $(i,j)$ of some image $I_k$, 

the mean $\mu$ across all pixels over all images is $$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$
And the pixel-wise mean $\mu_{ij}$ across all images is 
$$\mu_{ij}=\frac{1}{n}\sum_{k=1}^np_{ij}^{(k)}.$$
The general standard deviation $\sigma$ and pixel-wise standard deviation $\sigma_{ij}$ is defined similarly.

Which of the following preprocessing steps will not change the performance of a Nearest Neighbor classifier that uses L1 distance? Select all that apply.
1. Subtracting the mean $\mu$ ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
2. Subtracting the per pixel mean $\mu_{ij}$  ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu_{ij}$.)
3. Subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$.
4. Subtracting the pixel-wise mean $\mu_{ij}$ and dividing by the pixel-wise standard deviation $\sigma_{ij}$.
5. Rotating the coordinate axes of the data.

$\color{blue}{\textit Your Answer:}$1, 2


$\color{blue}{\textit Your Explanation:}$ L1 distance is abs($p_{ij}^{(k)}$-$p_{ij}^{(k')}$). Dividing by standard deviation will change the value, and rotating the coordinate axes will also change the value. But subtracting the means will not.

**Inline Question 3**

Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.
1. The decision boundary of the k-NN classifier is linear.
2. The training error of a 1-NN will always be lower than that of 5-NN.
3. The test error of a 1-NN will always be lower than that of a 5-NN.
4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
5. None of the above.

$\color{blue}{\textit Your Answer:}$4


$\color{blue}{\textit Your Explanation:}$The decision boundary of the k-NN classifier is obviously non-linear. The training error of a 1-NN may be higher than that of 5-NN, as shown in the plot above. The test error may be lower or may be not lower than that of a 5-NN. The time needed to classify a test example surely grows with the size of the training set, for it contains the comparison and calculations between the text example and every sample of the training set.

总结：

1.在kNN（以及其他算法中），涉及到大宗数据计算的，矩阵化处理比for循环快得多（快200多倍），哪怕只有一个for循环都会显著拖累速度。

2.kNN实在是太菜了，这就是个介绍而已。


二、
