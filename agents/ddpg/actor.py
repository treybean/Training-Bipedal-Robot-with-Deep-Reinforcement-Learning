from keras import layers, models, optimizers, regularizers
from keras import backend as K


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, env):
        """Initialize parameters and build model."""

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name="states")

        # Add hidden layers
        # net = layers.Dense(units=32)(states)
        # net = layers.BatchNormalization()(net)
        # net = layers.Activation("relu")(net)
        # net = layers.Dense(units=64)(net)
        # net = layers.BatchNormalization()(net)
        # net = layers.Activation("relu")(net)
        # net = layers.Dense(units=32)(net)
        # net = layers.BatchNormalization()(net)
        # net = layers.Activation("relu")(net)

        net = layers.Dense(units=400)(states)
        # net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=300)(net)
        # net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # # # Add final output layer with sigmoid activation
        # raw_actions = layers.Dense(
        #     units=self.action_size,
        #     activation="sigmoid",
        #     name="raw_actions",
        #     kernel_initializer=layers.initializers.RandomUniform(
        #         minval=-0.003, maxval=0.003
        #     ),
        # )(net)

        # # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # # Add final output layer with sigmoid activation
        # # raw_actions = layers.Dense(
        # #     units=self.action_size, activation="sigmoid", name="raw_actions"
        # # )(net)

        # # Scale [0, 1] output for each action dimension to proper range
        # # Create temp variable for storing self variables before using in lambda
        # # Without this, you get recursion error when saving Keras model: https://github.com/keras-team/keras/issues/12081
        # action_range = self.action_range
        # action_low = self.action_low

        # actions = layers.Lambda(
        #     lambda x: (x * action_range) + action_low, name="actions"
        # )(raw_actions)

        # Use tanh activation function with no scaling, since actions are in [-1,1]
        actions = layers.Dense(
            units=self.action_size,
            activation="tanh",
            name="actions",
            kernel_initializer=layers.initializers.RandomUniform(
                minval=-0.003, maxval=0.003
            ),
        )(net)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=1e-4)
        updates_op = optimizer.get_updates(
            params=self.model.trainable_weights, loss=loss
        )

        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[loss],
            updates=updates_op,
        )
