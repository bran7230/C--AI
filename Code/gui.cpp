#include <iostream>
#include <cmath>
#include <vector>
#include <windows.h>

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    
    case WM_COMMAND:
    if (LOWORD(wParam) == 1) // Button ID
    {
        char buffer[256];
        GetWindowText(GetDlgItem(hwnd, 2), buffer, 256); // Read text from input box
        std::string userInput = buffer;
        std::string response = "You said: " + userInput;
        MessageBox(hwnd, response.c_str(), "Ada Responds", MB_OK);
       
    }
    break;
}

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void LaunchGui(HINSTANCE hInstance, int nCmdShow)
{

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "Mainwin";
    wc.hbrBackground = CreateSolidBrush(RGB(34, 35, 37));

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0,
        "Mainwin",
        "Test window",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 
        NULL, NULL, hInstance, NULL);


    HWND button = CreateWindow(
        "BUTTON", //class name
        "Ask Ada", // text
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON ,
        600, 400, 300, 30, // x, y, width, height
        hwnd, //parent window
        (HMENU)1, //control id for message handling
        hInstance, NULL
    );

    
    HWND inputbox = CreateWindow(
        "EDIT",
        "",
        WS_VISIBLE | WS_CHILD | WS_BORDER | ES_LEFT,
        600, 350, 300, 25,         // x, y, width, height
        hwnd,                    // Parent window
        (HMENU)2,                // Control ID for this input
        hInstance, NULL
    );

        
    ShowWindow(hwnd, nCmdShow);

    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}
