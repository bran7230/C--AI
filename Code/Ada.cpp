#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include "gui.cpp"
#include <windows.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cstdlib>

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

std::string text;
std::unordered_map<char, int> chartoId;
std::unordered_map<int, char> idtoChar;
std::vector<int> inputs;
std::vector<int> targets;

int main()
{
    std::ifstream file("input.txt");
    if (!file.is_open())
    {
        std::cerr << "Failed to open input.txt" << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();
    
    int nextId = 0;
    std::vector<int> encodedText;
    for (char ch : text)
    {

        if (chartoId.find(ch) == chartoId.end())
        {
            chartoId[ch] = nextId;
            idtoChar[nextId] = ch;
            nextId++;
            encodedText.push_back(chartoId[ch]);
        }
    }
    for (size_t i = 0; i < encodedText.size() - 1; i++)
    {
        inputs.push_back(encodedText[i]);
        targets.push_back(encodedText[i + 1]);
    }

    std::cout << "\nTraining Pairs:\n";
    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::cout << inputs[i] << " -> " << targets[i] << std::endl;
    }

    return 0;
}
