#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-05 16:33:19
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-05 23:22:58
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/DCNN-MF-SP/Networks/ViT.py
Description: Modify here please
Init from https://github.com/Oussamayousre/automatic-classification-of-common-maternal-fetal-ultrasound-planes b784f0107fd8cd0368622c5da09a0b41d0a3eb04
'''
import tensorflow as tf
from tensorflow.keras import layers


# Implement multilayer perceptron (MLP).
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    return x


# Implement patch creation as a layer.
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches


# Implement the patch encoding layer.
# The PatchEncoder layer will linearly transform a patch by projecting it into a vector of size projection_dim.
# In addition, it adds a learnable position embedding to the projected vector.
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)

        return encoded


# Build the ViT model.
def ViT(model_name='ViT', input_channels=3, img_size=(256, 256), cls_num=6, pretrained=False,
        patch_size=6, # Size of the patches to be extract from the input images.
        projection_dim=64,
        num_heads=8,
        transformer_layers=4,
        mlp_head_units=[64, 32],  # Size of the dense layers of the final classifier
        ):
    num_patches = (img_size[0] // patch_size) ** 2

    transformer_units = [projection_dim * 2, projection_dim]  # Size of the transformer layers.

    inputs = layers.Input(shape=(img_size[0], img_size[1], input_channels))
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(cls_num, activation='softmax')(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    
    model.summary()

    return model    


if __name__ == '__main__':
    ViT(model_name='ViT', input_channels=3, img_size=(256, 256), cls_num=6)
