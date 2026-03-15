#include <opencv2/opencv.hpp>
#include <iostream>

// Declarations from glu.cpp
void glu_self_upsampling();
void glu_guided_upsampling();

int main(int argc, char** argv)
{
    // Uncomment to test self upsampling:
    glu_self_upsampling();

    // Test guided upsampling (default)
    glu_guided_upsampling();
    
    return 0;
}