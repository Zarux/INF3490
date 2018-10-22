"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np


class Neuron:
    k = 1

    def __init__(self, weights, value=0):
        self.value = value
        self.weights = [w for w in weights]

    @staticmethod
    def activate(x):
        k = Neuron.k
        return 1.0 / (1.0 + np.exp(-k * x))

    @staticmethod
    def d_activate(x):
        k = Neuron.k
        return (k * np.exp(-k * x)) / ((np.exp(-k * x) + 1)**2)


class Layer:
    def __init__(self, neurons, bias=None):
        self.neurons = neurons
        self.bias = bias

    def reset(self):
        for i, n in enumerate(self.neurons):
            self.neurons[i].value = 0


class MLP:
    def __init__(self, train, hidden, train_targets):
        self.beta = 1
        self.eta = 0.1

        self.layers = [
            Layer([Neuron(np.random.uniform(size=(hidden+1, )), value=t) for t in train[0]], Neuron(np.random.uniform(size=(hidden, )), value=-1)),
            Layer([Neuron(np.random.uniform(size=(len(train_targets[0]+1), ))) for _ in range(hidden)], Neuron(np.random.uniform(size=(len(train_targets[0]), )), value=-1)),
            Layer([Neuron([]) for _ in train_targets[0]])
        ]

    def earlystopping(self, inputs, targets, valid, validtargets, iterations=100):
        accuracies = []
        for it in range(iterations):
            data = list(zip(inputs, targets))
            np.random.shuffle(data)
            for i, t in data:
                new_input_layer = []
                for idx, neuron in enumerate(self.layers[0].neurons):
                    new_input_layer.append(Neuron(neuron.weights.copy(), value=i[idx]))
                self.layers[0] = Layer(new_input_layer, self.layers[0].bias)
                self.train(t)

            # Check errors after every epoch
            validation_data = list(zip(valid, validtargets))
            np.random.shuffle(data)
            output_list = []
            target_list = []
            for v, t in validation_data:
                new_input_layer = []
                for idx, neuron in enumerate(self.layers[0].neurons):
                    new_input_layer.append(Neuron(neuron.weights.copy(), value=v[idx]))
                self.layers[0] = Layer(new_input_layer, self.layers[0].bias)
                self.forward()
                o_vals = [n.value for n in self.layers[-1].neurons]
                output_list.append(o_vals.index(max(o_vals)))
                target_list.append(list(t).index(max(list(t))))

            classes = 8
            matrix = [[0 for _ in range(classes)] for _ in range(classes)]
            for out, tar in zip(output_list, target_list):
                matrix[tar][out] += 1
            accuracies.append(self.accuracy(matrix))

            # Detect drop in accuracy
            if len(accuracies) > 10 and np.mean(accuracies[-5:]) < np.mean(accuracies[-10:-5]):
                break

    def train(self, targets):
        deltas = [[], []]
        self.forward()

        # Deltas for output layer
        for idx, neuron in enumerate(self.layers[-1].neurons):
            deltas[0].append((neuron.value - targets[idx]) * Neuron.d_activate(neuron.value))

        # Deltas for hidden layer
        for idx, neuron in enumerate(self.layers[-2].neurons):
            delta_sum = 0
            for i, n in enumerate(self.layers[-1].neurons):
                delta_sum += deltas[0][i] * neuron.weights[i]
            deltas[1].append(delta_sum * Neuron.d_activate(neuron.value))

        # Hidden layer weights
        for idx, neuron in enumerate([self.layers[-2].bias] + self.layers[-2].neurons):
            for i, d in enumerate(deltas[0]):
                neuron.weights[i] = neuron.weights[i] - self.eta * d * neuron.value

        # Input layer weights
        for idx, neuron in enumerate([self.layers[-3].bias] + self.layers[-3].neurons):
            for i, d in enumerate(deltas[1]):
                neuron.weights[i] = neuron.weights[i] - self.eta * d * neuron.value

    def forward(self):

        for nidx, next_neuron in enumerate(self.layers[1].neurons):
            next_neuron_value = 0
            for idx, neuron in enumerate([self.layers[0].bias] + self.layers[0].neurons):
                next_neuron_value += neuron.value * neuron.weights[nidx]
            next_neuron.value = Neuron.activate(next_neuron_value)

        for nidx, next_neuron in enumerate(self.layers[2].neurons):
            next_neuron_value = 0
            for idx, neuron in enumerate([self.layers[1].bias] + self.layers[1].neurons):
                next_neuron_value += neuron.value * neuron.weights[nidx]
            next_neuron.value = Neuron.activate(next_neuron_value)

    @staticmethod
    def accuracy(cm):
        return sum(cm[i][i] for i in range(len(cm))) / sum([sum(col) for col in cm])

    def confusion(self, inputs, targets):
        output_list = []
        target_list = []
        for idx, i in enumerate(inputs):
            new_input_layer = []
            for idx2, neuron in enumerate(self.layers[0].neurons):
                new_input_layer.append(Neuron(neuron.weights, value=i[idx2]))
            self.layers[0] = Layer(new_input_layer, self.layers[0].bias)
            self.forward()
            o_vals = [n.value for n in self.layers[-1].neurons]
            output_list.append(o_vals.index(max(o_vals)))
            target_list.append(list(targets[idx]).index(max(list(targets[idx]))))

        classes = 8
        matrix = [[0 for _ in range(classes)] for _ in range(classes)]
        for out, tar in zip(output_list, target_list):
            matrix[out][tar] += 1

        print("   " + "  ".join([str(x) for x in range(classes)]))

        for idx, c in enumerate(matrix):
            print(idx, end="  ")
            for r in c:
                print("{:02}".format(r), end=" ")
            print()
        print("Model accuracy: {} ".format(self.accuracy(matrix)))


