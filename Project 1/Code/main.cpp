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
std::vector<cv::Mat> task2(cv::Mat image, const char* filename) {

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
    return {blue_image, green_image, red_image};
}


// Task 3: Display grayscale image of input image based on averaging the 3 color channels
cv::Mat task3(cv::Mat image){
    // Define height and width from input image
    int height = image.rows;
    int width = image.cols;
    float average;
    uchar grayscale_pixel;

    // Create grayscale image (Basically 1D array)
    cv::Mat gray;
    gray = cv::Mat::zeros(height, width, CV_8UC1); // Init to zeroes

    // Iterate through the image
    for (int y = 0; y < height; y++) { // Rows first
        for (int x = 0; x < width; x++) { // Columns then
            // Get Pixel at image
            cv::Vec3b pixel = image.at<cv::Vec3b>(y,x);
            average = (pixel[0] + pixel[1] + pixel[2])/3.0f; // Compute average
            grayscale_pixel = static_cast<uchar>(average); // Convert to proper type
            // Store it in the grapspace image
            gray.at<uchar>(y,x) = grayscale_pixel;
        }
    }

    printf("Task 3 complete, please enter Q to exit");

    // Create and display a window
    cv::namedWindow("Grayscale", cv::WINDOW_AUTOSIZE);
    cv::imshow("Grayscale", gray);
    cv::waitKey(0);
    cv::imwrite("output/grayscale.jpg", gray);

    return gray;
}

// Task 4 (Get and display the histogram of an image)
void task4(const cv::Mat& image, const std::string& name, const cv::Scalar& histColor){
    int hist[256] = {0};
    int height = image.rows;
    int width = image.cols;

    // Build histogram
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int val = static_cast<int>(image.at<uchar>(y,x));
            hist[val]++;
        }
    }

    // Margins for axes and labels
    int left_margin = 50;
    int bottom_margin = 40;
    int top_margin = 20;
    int right_margin = 20;

    // Create image for histogram (now 3-channel for color)
    int hist_w = 512, hist_h = 400;
    int canvas_w = hist_w + left_margin + right_margin;
    int canvas_h = hist_h + top_margin + bottom_margin;
    int bin_w = cvRound((double) hist_w / 256);
    cv::Mat histImage(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));

    // Normalize histogram
    int max_val = *std::max_element(hist, hist + 256);
    for(int i = 0; i < 256; i++) {
        hist[i] = static_cast<int>(((double)hist[i] / max_val) * hist_h);
    }

    // Draw histogram (start at left_margin, leave space at bottom for x-axis)
    for(int i = 1; i < 256; i++) {
        cv::line(histImage,
                 cv::Point(left_margin + bin_w * (i - 1), canvas_h - bottom_margin - hist[i - 1]),
                 cv::Point(left_margin + bin_w * i, canvas_h - bottom_margin - hist[i]),
                 histColor, 2, 8, 0);
    }

    // Draw y-axis (black)
    cv::line(histImage, cv::Point(left_margin, canvas_h - bottom_margin), cv::Point(left_margin, top_margin), cv::Scalar(0,0,0), 2);
    // Draw x-axis (black)
    cv::line(histImage, cv::Point(left_margin, canvas_h - bottom_margin), cv::Point(canvas_w - right_margin, canvas_h - bottom_margin), cv::Scalar(0,0,0), 2);

    // Add y-axis labels (0, max/2, max) in black
    cv::putText(histImage, "0", cv::Point(5, canvas_h - bottom_margin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    cv::putText(histImage, std::to_string(max_val/2), cv::Point(5, canvas_h - bottom_margin - hist_h/2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    cv::putText(histImage, std::to_string(max_val), cv::Point(5, top_margin + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);

    // Add x-axis labels (0, 128, 255) in black
    cv::putText(histImage, "0", cv::Point(left_margin, canvas_h - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    cv::putText(histImage, "128", cv::Point(left_margin + hist_w/2 - 10, canvas_h - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    cv::putText(histImage, "255", cv::Point(left_margin + hist_w - 20, canvas_h - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);

    cv::imshow((name + " Histogram").c_str(), histImage);
    cv::waitKey(0);
}

// Task 5: Perform Binary thresholding on the image
void task5(const cv::Mat& image) {
    // Declare a variable
    cv::Mat thresholded;
    
    // Get height and width
    int height = image.rows;
    int width = image.cols;
    int value;

    // Get user input
    std::cout<<"Enter the threshold (Task5) (between 0 and 255): ";
    std::cin>>value;

    thresholded = cv::Mat::zeros(height, width, CV_8UC1); // Init to zeroes

    // Threshold image
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if(static_cast<int>(image.at<uchar>(y,x)) < value) {
                thresholded.at<uchar>(y,x) = 0;
            }
            else {
                thresholded.at<uchar>(y,x) = image.at<uchar>(y,x);
            }
        }
    }

    // Display image
    // Create and display a window
    cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
    cv::imshow("thresholded", thresholded);
    cv::waitKey(0);
    cv::imwrite("output/thresholded.jpg", thresholded);

    
}

void task6(const cv::Mat& image) {
    int height = image.rows;
    int width = image.cols;
    int value;

    std::cout << "Enter Threshold Value(0-255): " << "\n";
    std::cin >> value;

    // Use int for gradients to avoid overflow/underflow
    cv::Mat gx = cv::Mat::zeros(height, width, CV_32S);
    cv::Mat gy = cv::Mat::zeros(height, width, CV_32S);
    cv::Mat gm = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat edge = cv::Mat::zeros(height, width, CV_8UC1);

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int cur = static_cast<int>(image.at<uchar>(y, x));
            int nextX = (x < width - 1) ? static_cast<int>(image.at<uchar>(y, x + 1)) : cur;
            int nextY = (y < height - 1) ? static_cast<int>(image.at<uchar>(y + 1, x)) : cur;

            gx.at<int>(y, x) = nextX - cur;
            gy.at<int>(y, x) = nextY - cur;

            float mag = std::sqrt(
                static_cast<float>(gx.at<int>(y, x) * gx.at<int>(y, x) +
                                   gy.at<int>(y, x) * gy.at<int>(y, x))
            );
            gm.at<float>(y, x) = mag;

            // Threshold to create edge image
            edge.at<uchar>(y, x) = (mag > value) ? 255 : 0;
        }
    }

    cv::namedWindow("Edge", cv::WINDOW_AUTOSIZE);
    cv::imshow("Edge", edge);
    cv::waitKey(0);
    cv::imwrite("output/edge.jpg", edge);
}


cv::Mat task7(const cv::Mat& image) {
    int height = image.rows;
    int width = image.cols;

    int heightNew = height / 2;
    int widthNew = width / 2;

    cv::Mat pyramid = cv::Mat::zeros(heightNew, widthNew, CV_8UC1);

    for(int y = 0; y < heightNew; y++) {
        for(int x = 0; x < widthNew; x++) {
            int cur    = static_cast<int>(image.at<uchar>(2*y,   2*x));
            int nextX  = static_cast<int>(image.at<uchar>(2*y,   2*x+1));
            int nextY  = static_cast<int>(image.at<uchar>(2*y+1, 2*x));
            int nextXY = static_cast<int>(image.at<uchar>(2*y+1, 2*x+1));
            int average = static_cast<int>((cur + nextX + nextY + nextXY) / 4.0f);
            pyramid.at<uchar>(y, x) = static_cast<uchar>(average);
        }
    }

    // Optional: Display or save the result
    cv::imshow("Pyramid", pyramid);
    cv::waitKey(0);
    cv::imwrite("output/pyramid.jpg", pyramid);

    return pyramid;
}


int main(int argc, char** argv) {
    // Create Variable for Storing Image and Further Processing
    cv::Mat image;
    cv::Mat grayscale;

    // Task 1 (Display Preset Image)
    image = task1("input/shed1-small.jpg");
    // End of Task 1

    //Task 2 (Isolate Color Channels)
    auto channels = task2(image, "shed1-small");
    // End of Task 2

    // Task 3 (Compute grayscale based on average)
    grayscale = task3(image);
    // End of Task 3

    // Task 4 (Histograms)
    std::vector<cv::Mat> blue_planes, green_planes, red_planes;
    cv::split(channels[0], blue_planes);   // blue_planes[0] is the blue channel
    cv::split(channels[1], green_planes);  // green_planes[1] is the green channel
    cv::split(channels[2], red_planes);    // red_planes[2] is the red channel

    // Now pass the single channel to task4:
    task4(blue_planes[0], "Blue", cv::Scalar(255,0,0));
    task4(green_planes[1], "Green", cv::Scalar(0,255,0));
    task4(red_planes[2], "Red", cv::Scalar(0,0,255));
    task4(grayscale, "Gray", cv::Scalar(0,0,0));
    // End of Task 4

    // Task 5 (Binary thresholding)
    task5(grayscale);
    // End of Task 5

    // Task 6 (Edge Finding)
    task6(grayscale);
    // End of Task 6

    // Task 7 (Pyramid Downsampling)
    auto ag2 = task7(grayscale);
    auto ag4 = task7(ag2);
    auto ag8 = task7(ag4);
    // End of Task 7

    return 0;
}