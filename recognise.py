import numpy
import scipy.special

char_number_map = {0:'0', 1:'1', 2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',
                   16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',
                   31:'V',32:'W',33:'X',34:'Y',35:'Z'}


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # number of input nodes
        self.input_nodes = inputnodes
        # number of hidden nodes
        self.hidden_nodes = hiddennodes
        # number of output nodes
        self.output_nodes = outputnodes
        # learning rate
        self.lr = learningrate

        self.weight_input_hidden = numpy.genfromtxt('input_to_hidden.csv', delimiter=',')

        self.weight_hidden_output = numpy.genfromtxt('hidden_to_output.csv', delimiter=',')

        # sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)


    def predict(self, inputs_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # inputs to hidden layer
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)

        # outputs from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # inputs to output layer
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)

        # outputs from the output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def recognize_single(rec_csv):

    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 36
    learning_rate = 0.005

    neural = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("Single data")
    test_data = open(rec_csv, 'r')
    test_list = test_data.readlines()
    test_data.close()


    for record in test_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[0:]) / 255 * 0.99) + 0.01
        outputs = neural.predict(inputs)
        label = numpy.argmax(outputs)
    print(label)
    print (char_number_map.get(label))
    return  char_number_map.get(label)


