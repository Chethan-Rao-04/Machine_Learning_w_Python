


def fetch_data(f_path):
    file_data=ny.genfromtxt(f_path, delimiter=",")
     # Convert the elements of variable file_data into array of float data type and round it.
    Data_FloatArray = ny.asarray(file_data, dtype=float)
    Data_FloatArray= ny.round(Data_FloatArray, decimals=5)

    # Extract X and Y and convert to float data type
    Y = Data_FloatArray[..., -1]
    X = Data_FloatArray[..., :-1]
    Y_float=Y.astype(float)
    X_float=X.astype(float)

    # Call the calculation funtion
    calc(X_float,Y_float,eta,iterations)
    

def sigmoid_fn(summation):
    return 1/(1+ny.exp(-summation))

def calc(X_float, Y_float, eta, iterations):

    # Hardcoding passive weights
    w_a_h1 = -0.3
    w_b_h1 = 0.4
    w_a_h2 = -0.1
    w_b_h2 = -0.4
    w_a_h3 = 0.2
    w_b_h3 = 0.1
    w_h1_o = 0.1
    w_h2_o = 0.3
    w_h3_o = -0.4
    w_bias_h1 = 0.2
    w_bias_h2 = -0.5
    w_bias_h3 = 0.3
    w_bias_o = -0.1

    for x in range(11):
        print('-', end=" ")

    weights_h1 = [w_bias_h1, w_a_h1, w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o]
    for weight in weights_h1:
        print(round(weight, 5), end=" ")

    print(round(w_h3_o, 5))

    

   
    for _ in range(iterations):
        for i in range(len(X_float)):
            
            
            # Summation of weights and inyuts including bias!
            summation_h1 = w_a_h1*X_float[i][0] + w_b_h1*X_float[i][1] + w_bias_h1 
            

            summation_h2 = w_a_h2*X_float[i][0] + w_b_h2*X_float[i][1] + w_bias_h2
            

            summation_h3 = w_a_h3*X_float[i][0] + w_b_h3*X_float[i][1] + w_bias_h3


            #Sigmoid Activation function!

            h1_output = sigmoid_fn(summation_h1)
            h2_output = sigmoid_fn(summation_h2)
            h3_output = sigmoid_fn(summation_h3)

            summation_net = h1_output *w_h1_o + h2_output*w_h2_o + h3_output*w_h3_o + w_bias_o

            net_out = 1/(1+ny.exp(-summation_net))

            #Calculating error by finding the difference between target and actual output!
            error = (Y_float[i]-net_out) 

            #Propagating the user backwards

            delta_o = net_out*(1-net_out)*error
            delta_h1 = h1_output*(1-h1_output)*(delta_o*w_h1_o)
            delta_h2 = h2_output*(1-h2_output)*(delta_o*w_h2_o)
            delta_h3 = h3_output*(1-h3_output)*(delta_o*w_h3_o)

            '''Weight update (Wji->Wji + delta(Wji)) 
                where delta(Wji)=learning rate * delta_o * Xji'''
            w_h1_o = w_h1_o + eta*delta_o*h1_output
            

            w_h2_o = w_h2_o + eta*delta_o*h2_output
            

            w_h3_o = w_h3_o + eta*delta_o*h3_output
            

            w_bias_o = w_bias_o + eta*delta_o

            #Similarly for hidden nodes.

            #h1
            w_a_h1 = w_a_h1 + eta*delta_h1*X_float[i][0]
            
            w_b_h1 = w_b_h1 + eta*delta_h1*X_float[i][1]
            
            w_bias_h1 = w_bias_h1 + eta*delta_h1
            
            
            #h2
            w_a_h2 = w_a_h2 + eta*delta_h2*X_float[i][0]
            
            w_b_h2 = w_b_h2 + eta*delta_h2*X_float[i][1]
            
            w_bias_h2 = w_bias_h2 + eta*delta_h2
            
            
            #h3
            w_a_h3 = w_a_h3 + eta*delta_h3*X_float[i][0]
            
            w_b_h3 = w_b_h3 + eta*delta_h3*X_float[i][1]
            
            w_bias_h3 = w_bias_h3 + eta*delta_h3

            #Printing all results
            print(X_float[i][0], end=" ")
            print(X_float[i][1], end=" ")
            outputs_list = [h1_output, h2_output, h3_output, net_out]
            for output in outputs_list:
                print(round(output, 5), end=" ")
            print(int(Y_float[i]), end=" ")

            values_list = [delta_h1, delta_h2, delta_h3, delta_o,w_bias_h1, w_a_h1, w_b_h1,w_bias_h2, 
                           w_a_h2, w_b_h2,w_bias_h3, w_a_h3, w_b_h3,w_bias_o,w_h1_o,w_h2_o]
            for value in values_list:
                print(round(value, 5), end=" ")
            print(round(w_h3_o, 5))
            
if __name__=="__main__":

    # to pass the command line arguments
    import argparse
    import numpy as ny
    argpar=argparse.ArgumentParser()
    argpar.add_argument("--data")
    argpar.add_argument("--eta")
    argpar.add_argument("--iterations")

    arguments= argpar.parse_args()
    file_path=arguments.data
    eta=arguments.eta
    eta=float(eta)
    iterations=arguments.iterations
    iterations = int(iterations)
    ds=fetch_data(file_path)