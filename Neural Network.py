import numpy as np
import nnfs
import pickle
import copy

nnfs.init()

# Layer
########################################################################################################################################################################################################################################

class Layer_Dense:
    def __init__(self, num_of_inputs:int, num_of_next_layer_neurons:int, weight_regularizer_L1:float = 0, weight_regularizer_L2:float = 0, bias_regularizer_L1:float = 0, bias_regularizer_L2:float = 0) -> None:
        self.weights = 1e-1 * np.random.randn(num_of_inputs, num_of_next_layer_neurons)
        self.biases = np.zeros((1, num_of_next_layer_neurons))
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)

        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
        
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0 ] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def get_paremeters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Layer_Dropout:
    def __init__(self, rate:float) -> None:
        self.rate = 1 - rate
    
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return 
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Layer_Input:
    def forward(self, inputs, training:bool):
        self.output = inputs

# Activation Function
########################################################################################################################################################################################################################################

class Activation_ReLU:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    def predictions(self, outputs):
        return outputs

class Activation_Linear:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    
    def predictions(self, outputs):
        return outputs

class Activation_Leaky_ReLU:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.maximum(inputs * 1e-2, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = -1e-2 * self.inputs[self.inputs < 0]
    
    def predictions(self, outputs):
        return outputs

class Activation_Parameterised_ReLU:
    def __init__(self, param:float) -> None:
        self.param = param
    
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.maximum(inputs * self.param, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = -self.param * self.inputs[self.inputs < 0]
    
    def predictions(self, outputs):
        return outputs

class Activation_ELU:
    def __init__(self, param:float = 1) -> None:
        self.param = param

    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.array([x if x >= 0 else self.param * (np.exp(x) - 1) for x in inputs])
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = self.param * np.exp(self.inputs[self.inputs < 0]) * dvalues[dvalues < 0]
    
    def predicitons(self, outputs):
        return outputs

class Activation_E_Swish:
    def __init__(self, param:float = 1) -> None:
        self.param = param

    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = self.param * inputs / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = (self.output + (1 / (1 + np.exp(-self.inputs))) * (1 - self.output)) * dvalues
    
    def predictions(self, outputs):
        return outputs

class Activation_Sign:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.sign(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = np.zeros_like(dvalues)
    
    def predictions(self, outputs):
        return outputs

class Activation_Soft_Sign:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = inputs / (1 + np.abs(inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues / (np.abs(self.inputs) + 1) ** 2
    
    def predictions(self, outputs):
        return outputs

class Activation_Binary_Step:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.array([0 if x <= 0 else 1 for x in inputs])
    
    def backward(self, dvalues):
        self.dinputs = np.zeros_like(dvalues)
    
    def predictions(self, outputs):
        return (outputs > 0) * 1

class Activation_Sigmoid:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Tanh:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.tanh(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues / np.cosh(self.inputs) ** 2
    
    def predictions(self, outputs):
        return (outputs > 0) * 1

class Activation_Hard_Tanh:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.maximum(-1, np.minimum(1, inputs))
    
    def backward(self, dvalues):
        self.dinputs = []
        for value in dvalues:
            if value < -1 or value >= 1:
                self.dinputs.append(0)
            else:
                self.dinputs.append(1)
        self.dinputs = np.array(self.dinputs) * dvalues
    
    def predictions(self, outputs):
        return (outputs > 0) * 1

class Activation_Softmax:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis = 1)

class Activation_Softplus:
    def forward(self, inputs, training:bool):
        self.inputs = inputs
        self.output = np.log(1 + np.exp(inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues / (1 + np.exp(-self.inputs))

# Loss Function
########################################################################################################################################################################################################################################

class Loss:
    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_L1 > 0:
                regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_L2 > 0:
                regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights ** 2)

            if layer.bias_regularizer_L1 > 0:
                regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_L2 > 0:
                regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases ** 2)
        return regularization_loss

    def remember_trainable_layers(self, trainable_layers:Layer_Dense or Layer_Dropout or Layer_Input):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, include_regularization:bool = False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization:bool = False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
    def new_pass(self):
        self.accumulated_sum, self.accumulated_count = 0, 0

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues / samples

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean(-(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)), axis = -1)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs / samples

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2, axis = -1)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs / samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis = -1)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs / samples

# Activation + Loss
########################################################################################################################################################################################################################################

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

# Accuracy
########################################################################################################################################################################################################################################

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    
    def calculated_accumulated(self):
        return self.accumulated_sum / self.accumulated_count

    def new_pass(self):
        self.accumulated_sum, self.accumulated_count = 0, 0 

class Accuracy_Regression(Accuracy):
    def __init__(self) -> None:
        self.precision = None
    
    def init(self, y, reinit:bool = False):
        if self.precision == None or reinit:
            self.precision = np.std(y) / 250
    
    def compare(self, predictions, y):
        return np.abs(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass
    
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y

    

# Optimizer
########################################################################################################################################################################################################################################

class Optimizer_SGD:
    def __init__(self, learning_rate:float = 1.0, decay:float = 0, momentum:float = 0) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iteration)

    def update_params(self, layer:Layer_Dense):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        self.iteration += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate:float = 1.0, decay:float = 0, epsilon:float = 1e-7) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iteration = 0
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iteration)
    
    def update_params(self, layer:Layer_Dense):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iteration += 1

class Optimizer_RMSporp:
    def __init__(self, learning_rate:float = 1e-3, decay:float = 0, epsilon:float = 1e-7, rho:float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iteration = 0
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iteration)
    
    def update_params(self, layer:Layer_Dense):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biase_cache = np.zeros_like(layer.biases)
        
        layer.weight_cahce = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cahce = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iteration += 1

class Optimizer_Adam:
    def __init__(self, learning_rate:float = 1e-3, decay:float = 0, epsilon:float = 1e-7, beta_1:float = 0.9, beta_2:float = 0.999) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iteration = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iteration)
    
    def update_params(self, layer:Layer_Dense):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iteration + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iteration + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iteration + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iteration + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
        self.iteration += 1

# Model
########################################################################################################################################################################################################################################

class Model:
    def __init__(self) -> None:
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer:Layer_Dense or Layer_Dropout):
        self.layers.append(layer)

    def add_many(self, layer_list:list):
        for layer in layer_list:
            self.layers.append(layer)
    
    def initialize(self, *, loss, optimizer, accuracy):
        if loss:
            self.loss = loss
        if optimizer:
            self.optimizer = optimizer
        if accuracy:
            self.accuracy = accuracy
    
    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        if self.loss:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, x, y, *, epochs:int = 1, batch_size:int = None, validation_data = None):
        self.accuracy.init(y)
        train_steps = 1
        if batch_size:
            train_steps = len(x) // batch_size
            if train_steps * batch_size < len(x):
                train_steps += 1
        for epoch in range(epochs):
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):
                if not batch_size:
                    batch_x = x
                    batch_y = y
                else:
                    batch_x = x[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]
                output = self.forward(x, training = True)
                data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization = True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y)
                self.backward(output, y)
                self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                print(f"Epoch: {epoch + 1}   Accuracy: {accuracy:.3f}   Loss: {loss:.3f}")
        epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization = True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()
        print(f"Accuracy: {epoch_accuracy:.3f}   Loss: {epoch_loss:.3f}")
        if validation_data:
            self.evaluate(*validation_data, batch_size = batch_size)

    def evaluate(self, x_val, y_val, *, batch_size:int = None):
        validation_steps = 1
        if batch_size:
            validation_steps = len(x_val) // batch_size
            if validation_steps * batch_size < len(x_val):
                validation_steps += 1
        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
            if not batch_size:
                batch_x = x_val
                batch_y = y_val
            else:
                batch_x = x_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]
            output = self.forward(batch_x, training = False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f"Accuracy: {validation_accuracy:.3f}   Loss: {validation_loss:.3f}")

    def forward(self, x, training:bool):
        self.input_layer.forward(x, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    
    def backward(self, output, y):
        if self.softmax_classifier_output:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def save_parameterss(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    
    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def predict(self, x, *, batch_size:int = None):
        predictions_steps = 1
        if batch_size:
            predictions_steps = len(x) // batch_size
            if predictions_steps * batch_size < len(x):
                predictions_steps += 1
        output = []
        for step in range(predictions_steps):
            if not batch_size:
                batch_x = x
            else:
                batch_x = x[step * batch_size:(step + 1) * batch_size]
            batch_output = self.forward(batch_x, training = False)
            output.append(batch_output)
        return np.vstack(output)
    
    @staticmethod
    def load(path):
        with open(path, 'wb') as f:
            model = pickle.load(f)
        return model

########################################################################################################################################################################################################################################
if __name__ == '__main__':
    pass
