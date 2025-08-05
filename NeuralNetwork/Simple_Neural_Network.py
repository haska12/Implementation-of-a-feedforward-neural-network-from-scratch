import numpy as np
from sklearn.preprocessing import OneHotEncoder
"""
this class is a simple neural network with  input_size input 1 hiden layer  of hidden_size number of neuron ,and output_size outpute
"""

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,activation_name='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name=activation_name
        

        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  
        return 1 / (1 + np.exp(-x))
    
    def Relu(self, x):
        x = np.clip(x, -500, 500)  
        return np.maximum(0, x)
    def sigmoid_derivative(self, x):
        x = np.clip(x, -500, 500)  
        s = self.sigmoid(x)
        return x * (1 - x)  

    def relu_derivative(self, x):
        x = np.clip(x, -500, 500)  
        return (x > 0).astype(float)
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def softmax_derivative(self,x):
        
        s = self.softmax(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def apply_activation(self, x):
        if self.activation_name == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_name == 'relu':
            return self.Relu(x)
        elif self.activation_name == 'softmax':
            return self.softmax(x)
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x):
        if self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_name == 'relu':
            return self.relu_derivative(x)
        elif self.activation_name == 'softmax':
            return self.softmax_derivative(x)
        else:
            raise ValueError("Derivative not supported for this activation.")

    def forward(self, x):

        hiden_layer_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden #z (hidden)= x(input vector) *w(hidden) +b(hidden)
        hidden_layer_output = self.apply_activation(hiden_layer_input) #a=f(z)

        output_layer_input=np.dot(hidden_layer_output,self.weights_hidden_output)+self.bias_output  #z (output) =z (huden)*w(output)+b(output) 
        predicted_value=self.apply_activation(output_layer_input) #a=f(z)

        return predicted_value,hidden_layer_output
    def backward(self,x,y,learn_rate,predicted_value,hidden_layer_output):
        
        output_error=y-predicted_value
        #function inGradient is mse 
        Gradient_output=output_error*self.activation_derivative(predicted_value)

        hidden_error=np.dot(Gradient_output,self.weights_hidden_output.T)
        Gradient_hidden=hidden_error*self.activation_derivative(hidden_layer_output)

        #update w
        self.weights_hidden_output+=np.dot(hidden_layer_output.T,Gradient_output)*learn_rate
        self.weights_input_hidden+=np.dot(x.T,Gradient_hidden)*learn_rate
        #update b
        self.bias_output+=np.sum(Gradient_output, axis=0,
                               keepdims=True) * learn_rate
        self.bias_hidden+= np.sum(Gradient_hidden, axis=0,
                               keepdims=True) * learn_rate
        

    def train(self,x,y,learn_rate=0.01,epoch=2000,limite_error=10e-5,desplay_fr=1000):
        for i in range(epoch):
            predicted_value,hidden_layer_output=self.forward(x)

            self.backward(x,y,learn_rate,predicted_value,hidden_layer_output)
            #print(predicted_value)
            loss = np.mean(np.square(y - predicted_value))
            if loss<limite_error:
                print(f"Training stopped early at epoch {i}, Loss: {loss}")
                print(f"Epoch {i}, Loss: {loss}")
                break
            if i % desplay_fr == 0:
                print(f"Epoch {i}, Loss: {loss}")
    def predict(self, x):
        pred, _ = self.forward(x)
        return np.argmax(pred, axis=1)

    def predict_proba(self, x):
        pred, _ = self.forward(x)
        return pred
    def MSE(sdelf,y,predicted_value):
        return np.sum((predicted_value-y)**2)*0.5

if __name__ == "__main__":
    #xor txt 
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
    y = np.array([[0], [1], [1], [0]])

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=2,activation_name='softmax')
    nn.train(X, y, epoch=3, learn_rate=0.1)
    X_trin = np.array([[5, 0], [0, 7], [8, 0], [8, 7]])  
    y_train = np.array([[0], [1], [1], [0]])

    output = nn.predict(X_trin)
    
    accuracy = np.mean(output == y_train)
    print("accuracy")

    print(accuracy)


    print("Predictions after training:")
    print(output)







