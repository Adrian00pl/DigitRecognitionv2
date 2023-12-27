import numpy as np

# Funkcja aktywacji - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji aktywacji - dla algorytmu wstecznej propagacji
def sigmoid_derivative(x):
    return x * (1 - x)

# Klasa reprezentująca sieć neuronową
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicjalizacja wag dla warstw ukrytej i wyjściowej
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def feedforward(self, inputs):
        # Propagacja w przód
        self.hidden = sigmoid(np.dot(inputs, self.weights_input_hidden))
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return self.output

    def backward(self, inputs, targets, learning_rate):
        # Obliczanie błędu dla warstwy wyjściowej i ukrytej
        output_error = targets - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Aktualizacja wag
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

    def train(self, training_inputs, training_targets, epochs, learning_rate):
        for epoch in range(epochs):
            for inputs, targets in zip(training_inputs, training_targets):
                inputs = inputs.reshape(1, -1)
                targets = targets.reshape(1, -1)
                self.feedforward(inputs)
                self.backward(inputs, targets, learning_rate)

    def predict(self, inputs):
        inputs = inputs.reshape(1, -1)
        return self.feedforward(inputs)


# Przykładowe dane uczące - reprezentacja liter A, B, C
# Każda litera jest zapisana jako 5x3 macierz, gdzie 1 oznacza aktywację neuronu, a 0 brak aktywacji
inputs = [
    np.array([[0, 1, 0],   # A
              [1, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [1, 0, 1]]),

    np.array([[1, 1, 0],   # B
              [1, 0, 1],
              [1, 1, 0],
              [1, 0, 1],
              [1, 1, 0]]),

    np.array([[0, 1, 1],   # C
              [1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 1, 1]])
]

# Oczekiwane wyjścia dla liter A, B, C
targets = [
    np.array([[1],
              [0],
              [0]]),

    np.array([[0],
              [1],
              [0]]),

    np.array([[0],
              [0],
              [1]])
]

# Inicjalizacja sieci neuronowej
input_size = 15  # Rozmiar wejścia - 5x3 dla każdej litery
hidden_size = 8   # Liczba neuronów w warstwie ukrytej
output_size = 1   # Liczba neuronów w warstwie wyjściowej

# Utworzenie instancji sieci neuronowej
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Uczenie sieci
nn.train(inputs, targets, epochs=10000, learning_rate=0.1)

# Testowanie sieci na przykładowych danych
test_input = np.array([[0, 1, 0],  # Test na literze A
                       [1, 0, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [1, 0, 1]])

prediction = nn.predict(test_input)
print("Prediction for letter A:")
print(prediction)  # Wyświetlenie przewidywanej wartości dla litery A
