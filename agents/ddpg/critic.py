from keras import layers, models, optimizers, regularizers
from keras import backend as K


class Critic:
    """Critic (Value) Model."""

    def __init__(self, env):
        """Initialize parameters and build model.
        """
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name="states")
        actions = layers.Input(shape=(self.action_size,), name="actions")

        states_and_actions = layers.concatenate([states, actions])
        net = layers.Dense(units=400, activation="relu")(states_and_actions)
        net = layers.Dense(units=300, activation="relu")(net)

        # Add hidden layer(s) for state pathway
        # net_states = layers.Dense(units=32, kernel_regularizer=regularizers.l2(0.01))(
        #     states
        # )
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Activation("relu")(net_states)
        # net_states = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.01))(
        #     net_states
        # )
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Activation("relu")(net_states)

        # # Add hidden layer(s) for action pathway
        # net_actions = layers.Dense(units=32, kernel_regularizer=regularizers.l2(0.01))(
        #     actions
        # )
        # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Activation("relu")(net_actions)
        # net_actions = layers.Dense(units=64, kernel_regularizer=regularizers.l2(0.01))(
        #     net_actions
        # )
        # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Activation("relu")(net_actions)

        # # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # # Combine state and action pathways
        # net = layers.Add()([net_states, net_actions])
        # net = layers.Activation("relu")(net)

        # # Add more layers to the combined network if needed

        # # Add final output layer to prduce action values (Q values)
        # Q_values = layers.Dense(units=1, name="q_values")(net)

        # net_states = layers.Dense(units=400)(states)
        # # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Activation("relu")(net_states)

        # net_states = layers.Dense(units=300)(net_states)
        # # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.Activation("relu")(net_states)

        # # # Add hidden layer(s) for action pathway
        # net_actions = layers.Dense(units=400)(actions)
        # # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Activation("relu")(net_actions)

        # net_actions = layers.Dense(units=300)(net_actions)
        # # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.Activation("relu")(net_actions)

        # # # Combine state and action pathways
        # net = layers.Add()([net_states, net_actions])
        # net = layers.Activation("relu")(net)

        # # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(
            units=1,
            name="q_values",
            kernel_initializer=layers.initializers.RandomUniform(
                minval=-0.003, maxval=0.003
            ),
        )(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=1e-3)
        self.model.compile(optimizer=optimizer, loss="mse")

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients
        )

