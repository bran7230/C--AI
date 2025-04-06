#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.cpp"
#include "gui.cpp"
#include <windows.h>

int main()
{ 
  
    HINSTANCE hInstance = GetModuleHandle(NULL);
    LaunchGui(hInstance, SW_SHOW);
 

    return 0;
}
