#include <iostream>
#include <cmath>
#include <vector>

//Relu math:
//I plan to upgrade this for Matrices in the Future.
std::vector<float> relu( const std::vector<float>& value) {
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
float sigmoid(float z){
    //Return Sigmoid function ( 1 / 1 + E ^-z)
    return 1.0 / (1.0 + std::exp(-z));
}


int main() {
    std::vector<float> test = {1.0f};
    std::vector<float> result = relu(test); 
    
    for( float val :result)
    {
        std::cout << val << " ";
        
    }
    std::cout<< "\nDone values." << std::endl;
    std::cout << sigmoid(2) << std::endl;
    return 0;
}


