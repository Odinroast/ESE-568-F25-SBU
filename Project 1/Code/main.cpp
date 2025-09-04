#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// Reads an image from file, displays it in a window, and returns the image matrix
cv::Mat task1(const char* file) {
    // Read the image file into a CV matrix
    cv::Mat image;
    image = cv::imread(file, cv::IMREAD_COLOR);
    
    // Check for failure (No image data)
    if(image.empty())
    {
        printf("No image data - OpenCV failed to load the image\n");
        printf("This could be due to unsupported format or corrupted file\n");
        exit(1);
    }
    
    printf("Image loaded successfully! Size: %dx%d\n", image.cols, image.rows);
    printf("Task 1 complete. Click Q to move on to the next one.\n");

    // Create window for display and display the image inside that window
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::waitKey(0);

    // Return the image matrix for potential further processing
    return image;
}

// Takes an image matrix and isolates the 3 color channels, displaying each one in a window and saving them to output/filename_color.jpg
void task2(cv::Mat image, const char* filename) {

    // Create 3-channel color images for each channel (BGR format)
    cv::Mat blue_image, green_image, red_image;

    // Define height and width from input image
    int height = image.rows;
    int width = image.cols;

    // Create 3-channel color images initialized to zero (black)
    blue_image = cv::Mat::zeros(height, width, CV_8UC3);
    green_image = cv::Mat::zeros(height, width, CV_8UC3);
    red_image = cv::Mat::zeros(height, width, CV_8UC3);

    // Iterate through each pixel and assign values to the respective channel matrices
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Get the pixel at (x, y)
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            
            // For blue image: only blue channel active, others zero
            blue_image.at<cv::Vec3b>(y, x)[0] = pixel[0]; // Blue
            blue_image.at<cv::Vec3b>(y, x)[1] = 0;        // Green = 0
            blue_image.at<cv::Vec3b>(y, x)[2] = 0;        // Red = 0
            
            // For green image: only green channel active, others zero
            green_image.at<cv::Vec3b>(y, x)[0] = 0;        // Blue = 0
            green_image.at<cv::Vec3b>(y, x)[1] = pixel[1]; // Green
            green_image.at<cv::Vec3b>(y, x)[2] = 0;        // Red = 0
            
            // For red image: only red channel active, others zero
            red_image.at<cv::Vec3b>(y, x)[0] = 0;        // Blue = 0
            red_image.at<cv::Vec3b>(y, x)[1] = 0;        // Green = 0
            red_image.at<cv::Vec3b>(y, x)[2] = pixel[2]; // Red
        }
    }

    // Create window and display blue channel, green channel and red channel
    cv::namedWindow("Blue Channel", cv::WINDOW_AUTOSIZE );
    cv::imshow("Blue Channel", blue_image);
    cv::waitKey(0);
    cv::namedWindow("Green Channel", cv::WINDOW_AUTOSIZE );
    cv::imshow("Green Channel", green_image);
    cv::waitKey(0);
    cv::namedWindow("Red Channel", cv::WINDOW_AUTOSIZE );
    cv::imshow("Red Channel", red_image);
    cv::waitKey(0);

    // Save the channel images to output/filename_color.jpg
    cv::imwrite(std::string("output/") + filename + "_blue.jpg", blue_image);
    cv::imwrite(std::string("output/") + filename + "_green.jpg", green_image);
    cv::imwrite(std::string("output/") + filename + "_red.jpg", red_image);
    
    printf("Task 2 complete. Channel images saved to output folder.\n");
}



int main(int argc, char** argv) {
    // Create Variable for Storing Image and Further Processing
    cv::Mat image;

    // Task 1 (Display Preset Image)
    image = task1("input/shed1-small.jpg");
    // End of Task 1

    //Task 2 (Isolate Color Channels)
    task2(image, "shed1-small");
    // End of Task 2

    
    return 0;
}