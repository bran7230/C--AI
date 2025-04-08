#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include "gui.cpp"
#include <windows.h>
#include <fstream>
// Use for gui tests
/*
HINSTANCE hInstance = GetModuleHandle(NULL);
LaunchGui(hInstance, SW_SHOW);
*/
// Use for files:
/*
std::ofstream MyFile("Adacode.txt");
    MyFile << "Test!" <<std::endl;

    MyFile.close();
*/
int main()
{   std::ofstream MyFile("Adacode.txt");
    MyFile << "Test!" <<std::endl;

    MyFile.close();
    return 0;
}
