from tensorflow import keras

class LoraLayer(keras.layers.Layer):
    def __init__(self, original_layer, rank=4, alpha=4., trainable=False, use_bias=False, **kwargs):
        # We want to keep the name of this layer the same as the original
        # dense layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)
        
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.original_layer = original_layer
        self.original_layer.trainable = False
        
        self._num_heads = original_layer_config["output_shape"][-2]
        self._hidden_dim = self._num_heads * original_layer_config["output_shape"][-1]

        self.A = keras.layers.Dense(
            units=rank,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.RandomNormal(stddev=1 / self.rank),
            trainable=trainable,
            name="lora_a"
        )

        self.B = keras.layers.EinsumDense(
            equation=original_layer_config["equation"],
            output_shape=original_layer_config["output_shape"],
            kernel_initializer="zeros",
            trainable=trainable,
            name=f"lora_B",
        )
    def call(self, inputs):
        original_output = self.original_layer(inputs)
        lora_output = self.B(self.A(inputs)) * self.scale
        return original_output + lora_output