import os
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 123

#读取训练集
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

#读取验证集
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


#读取测试集
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

#取出类别名和类别数
class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

#优化数据读取速度
"""
cache()
把数据缓存起来，避免每轮 epoch 都重新从磁盘慢慢读。

shuffle(1000)
把训练数据打乱。
这是很重要的，因为如果图片按类别顺序排着，模型容易学偏。
只给训练集 shuffle，不给验证集 shuffle，因为验证集只需要稳定评估。

prefetch(...)
让 CPU 提前准备下一批数据，模型训练时不用一直等。

AUTOTUNE
TensorFlow 自动帮你选一个合适的预取策略。
"""
AUTOTUNE = tf.data.AUTOTUNE 
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)



#数据增强
"""
这一步是在训练时“随机改造图片”，让模型更鲁棒。

RandomFlip("horizontal")
随机水平翻转图片。

RandomRotation(0.1)
随机旋转一点点。

作用是：
增加数据多样性
减少过拟合
让模型不要死记硬背某些固定角度

注意：
这些增强通常只在训练时生效，验证时不会乱转。
"""
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)), #输入归一化
    data_augmentation,  #数据增强层(在训练前就做数据增强)

    #3*3的卷积核，有32个，所以会生成32个feature map
    #ReLu: f(x)=max(0,x)
    #输出尺寸=输入尺寸−kernel尺寸+1
    #输入尺寸：(224, 224, 3) 输出尺寸：(222, 222, 32)
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    #MaxPooling2D(pool_size=(2,2)) 用一个 2×2 的窗口 在特征图上滑动，每次只保留最大值
    #输出尺寸：(111,111,32)
    tf.keras.layers.MaxPooling2D(),

    #卷积核尺寸：3 × 3 × 32
    #输入:  (111, 111, 32) 输出:  (109, 109, 64)
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    #输入:  (109, 109, 64) 输出:  (54, 54, 64)
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"), #这是一个全连接层，把 Flatten 后的高维特征进一步组合，压缩成 128 个更抽象的特征。
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam", #它负责根据 loss 反向传播，更新权重
    loss="sparse_categorical_crossentropy", #这是多分类任务常用损失函数（标签是 one-hot 向量）
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig("outputs/figures/baseline_accuracy.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("outputs/figures/baseline_loss.png")
plt.close()

model.save("outputs/checkpoints/baseline_cnn.keras")

print("Training finished.")
print("Model saved to outputs/checkpoints/baseline_cnn.keras")