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
        MessageBox(hwnd, "Hi, I'm Ada. How can I help?", "Ada", MB_OK | MB_ICONINFORMATION);
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
    wc.hbrBackground = CreateSolidBrush(RGB(173, 216, 230));

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
        WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        20, 20, 100, 30, // x, y, width, height
        hwnd, //parent window
        (HMENU)1, //control id for message handling
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
