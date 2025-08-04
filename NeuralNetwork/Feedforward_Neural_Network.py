import numpy as np

class NeuralNetwork:
    def __init__(self, input_size,hidden_layers, output_size,activation_hidden='relu',activation_output='softmax'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_hidden_name=activation_hidden
        self.activation_output_name=activation_output
        # Sizes: [input size , hidden1 sizes, hidden2, ..., output size]
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(layer_sizes) - 1

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1. / layer_sizes[i])#to make w shour thar defrent in w note big
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)

            self.biases.append(b)

        
        
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  
        return 1 / (1 + np.exp(-x))
    
    def Relu(self, x):
        x = np.clip(x, -500, 500)  
        return np.maximum(0, x)
    def sigmoid_derivative(self, x_activated): 
        return x_activated * (1 - x_activated)

    def relu_derivative(self, x_activated):
        return (x_activated > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def softmax_derivative(self,x):
        
        s = self.softmax(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def apply_activation(self, x,activation_name='sigmoid'):
        if activation_name == 'sigmoid':
            return self.sigmoid(x)
        elif activation_name == 'relu':
            return self.Relu(x)
        elif activation_name == 'softmax':
            return self.softmax(x)
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x,activation_name='sigmoid'):
        if activation_name == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif activation_name == 'relu':
            return self.relu_derivative(x)
        elif activation_name == 'softmax':
            return self.softmax_derivative(x)
        else:
            raise ValueError("Derivative not supported for this activation.")


    def forward(self, x):
        activations = [x]  #value of output of each neuron after activations function
        pre_activations = [] #value of output of each neuron before activations function
        for i in range(self.num_layers):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]  # z=w *a +b
            pre_activations.append(z) 
            # use corect activation  function for each layer
            if  i == self.num_layers - 1:
                a = self.apply_activation(z,self.activation_output_name)
            else:
                a= self.apply_activation(z,self.activation_hidden_name)
            activations.append(a)
        #activations list output for ech layer include oupute
        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true, learn_rate):
        num_samples = y_true.shape[0] 
        delta = activations[-1] - y_true #error output
        #Loop over layers in reverse 
        for i in range(self.num_layers - 1, -1, -1): 
            dW = np.dot(activations[i].T, delta) / num_samples #Gradient dW = ∂L/∂W[i]/ dL/dW = (input^T) × (error) input=activations[i]  delta=error
            db = np.sum(delta, axis=0, keepdims=True) / num_samples #Gradient db =
            self.weights[i] -= learn_rate * dW #update weights w new=  w old - l *dw
            self.biases[i] -= learn_rate * db #update biases b new= b old - l* db

            # calcul delta for non input/ calcul delta for next 
            if i > 0: 
                delta_propagated_to_prev_layer = np.dot(delta, self.weights[i].T)
                deriv_activation_name = self.activation_hidden_name
                activation_deriv_val = self.activation_derivative(activations[i], deriv_activation_name)
                delta = delta_propagated_to_prev_layer * activation_deriv_val #Propagate error backward delta new= (delta old * W.T) * f'(z)	
    def categorical_cross_entropy(self, y_true, y_pred, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)
    def train(self, x, y, learn_rate=0.01,epoch=1000,limite_error=10e-5,desplay_fr=1000):
        loss_history = [] 
        for i in range(epoch):
            activations, pre_activations = self.forward(x)

            self.backward(activations, pre_activations, y,learn_rate=learn_rate)
  
            pred = np.argmax(activations[-1], axis=1)
            loss = self.categorical_cross_entropy(y, activations[-1])
            loss_history.append(loss)
            
            if loss<limite_error:
                print(f"Training stopped early at epoch {i}, Loss: {loss}")
                print(f"Epoch {i}, Loss: {loss}")
                break
            if i % desplay_fr == 0:
                print(f"Epoch {i}, Loss: {loss}")
        return loss_history
    def predict(self, x):
        activations, _ = self.forward(x)
        return np.argmax(activations[-1], axis=1), activations[-1]
    def predict_proba(self, x):
        activations, _ = self.forward(x)
        return activations[-1]
    def MSE(sdelf,y,predicted_value):
        return np.sum((predicted_value-y)**2)*0.5
    
if __name__ == "__main__":
    #xor txt 
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_size=2, hidden_layers=[4,16], output_size=2)  #  2 layers 4' size of 1 hidden layer 16 size of 2 hidden layer 
    loss_history=nn.train(X, y, epoch=10000, learn_rate=0.001,desplay_fr=500)

    
    X_test = np.array([[5, 0], [0, 7], [8, 0], [8, 7]])
    y_test = np.array([[0], [1], [1], [0]])

    pred_labels, probs = nn.predict(X_test)
    print(probs)
    accuracy = np.mean(pred_labels.reshape(-1, 1) == y_test)
    print("Predictions:", pred_labels)
    print("Accuracy:", accuracy)

