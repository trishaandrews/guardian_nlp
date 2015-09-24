from pyfann import libfann
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#create a new FANN neural network with the given number of input and output nodes, randomizes the initial weights, sets the learning rate, and the activation function, returns the network
def create_network(input_nodes, output_nodes):
    ann = libfann.neural_net()
    if input_nodes > 6:
        hidden_nodes = int((input_nodes + output_nodes)/2)
    else:
        hidden_nodes = 2
    layers = [input_nodes, hidden_nodes, output_nodes]
    #ann.create_standard_array(layers)
    ann.create_sparse_array(0.7, layers)
    ann.randomize_weights(-0.5, 0.5)
    ann.set_learning_rate(0.6)
    ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)  
    return ann
    
#trains the network on a given file for a specificed number of epochs, or until the desired error value is reached, saves the network to the specified file for later use every few epochs
def train_network(data_filename, ann, max_num_epochs, print_freq, des_err, network_filename, X_test, y_test): 
    for i in range(100):
        print i
        y_pred = []
        
        ann.train_on_file(data_filename, max_num_epochs, print_freq, des_err)	
        ann.save(network_filename + str(i) + ".net")
        y_pred = ann_predict(ann, X_test)
        prec, recall, fscore, sup = precision_recall_fscore_support(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print prec
        print recall
        print acc
   
def get_test_data(datafilename):
    X_test = []
    y_test = []
    with open(datafilename, "r") as f:
        line = f.readline() #parameter line
        line = f.readline() #first input line
        while line:
            inputs = []
            tokens = line.split()
            inputs = [float(t.strip()) for t in tokens]
            
            line = f.readline() #classlabel line
            labeltokens = line.split()
            y_t = [float(l.strip()) for l in labeltokens]

            if len(y_t) > 0:
                y_ts = y_t.index(max(y_t)) + 1
                y_test.append(y_ts)
                X_test.append(inputs)
            f.readline() #next input
    return X_test, y_test

def ann_predict(ann, X_test):
    y_pred = []
    for xt in X_test:
        yp = ann.run(xt)
        maxout = 0.0
        maxind = -1
        for i, lab in enumerate(yp):
            if lab > maxout:
                maxout = lab
                maxind = i
        y_p = maxind + 1
        #print y_t, y_ts
        #print outputs, y_p
        y_pred.append(y_p)
    return y_pred

#y_test = []
#X_test = []

topic = "arts"
r_state = "8"

X_test, y_test = get_test_data("ann_test_" + topic + "_" +r_state + ".txt")  
#run the program, network values correspond to inputs and outputs
ann = create_network(166, 4) #number inputs, outputs
train_network("ann_train_" + topic + "_" + r_state +".txt", ann, 1, 1, 0.0001, "./nets/trainednetwork_", X_test, y_test)
#test_vec is a properly formatted input file header of number data, inputs, outputs, one
#input per line after
