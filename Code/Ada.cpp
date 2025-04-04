#include <iostream>
#include <cmath>
#include <vector>


//TODO: upgrade this for Matrices in the Future.
// Applies ReLU to a vector of floats. Returns a vector of activated values.
std::vector<float> relu( const std::vector<float>& value) {
    //Example running:
    /*
    std::vector<float> test = {1.0f};
    std::vector<float> result = relu(test); 
    for( float val :result)
    {
        std::cout << val << " ";
        
    }
    std::cout<< "\nDone values." << std::endl;
    */

    //vectorize input or value:
    std::vector<float> output; 
    //reserve memory or space for perfomance
    output.reserve(value.size()); 

    //loop through values(for matrixes ex 128 would be quicker than manual check)
    for(float val : value)
    {
        //If value is greater than 0, add to vector, else, return 0
        output.push_back( val < 0.0f ? 0 : val);
    }

    return output; 
    
}

//Confidence or sigmoid function
//Added vectors
std::vector<float> sigmoid(const std::vector<float>& z){
    //Example running:
    /*
    std::vector<float> testsig = {1.0f, 2.0f, -1.0f};
    std::vector<float> resig = sigmoid(testsig);
     for(float val: resig)
    {
        std::cout << val << " ";
    }
    */

    //Return Sigmoid function ( 1 / 1 + E ^-z)
    std::vector<float> output;
    output.reserve(z.size());

    //Loop through values and add Sigmoid vals
    for(float val : z){
        output.push_back( 1 / (1 + (std::exp(-val))));
    }
    return output;
}


int main() {
    std::vector<float> test = {1.0f};
    std::vector<float> result = relu(test); 
    for( float val :result)
    {
        std::cout << val << " ";
        
    }
    std::cout<< "\nDone values." << std::endl;
   
    return 0;
}


