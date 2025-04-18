#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include <sstream> 
#include <fstream>
#include <string>
#include <unordered_map>





int main()
{

    std::vector<float> probs = {0.1, 0.7, 0.2};
    int target = 1; 
    std::vector<float> dz = computeGradient(probs, target);
    for(float val : dz)
    {
        std::cout<<val <<std::endl;
    }

    
    return 0;
}
