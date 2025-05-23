#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

Mat loadImage(const string& path);
Mat convertToYCrCb(const Mat& src);
Mat segmentSkin(const Mat& ycrcb);
Mat cleanMask(const Mat& mask);
vector<vector<Point>> findSkinContours(const Mat& mask);
Rect getLargestBoundingRect(const vector<vector<Point>>& contours);
Mat createEyeMask(const Mat& faceROI);
vector<Rect> detectEyes(const Mat& eyeMask);
void drawRectangleOnImage(Mat& img, const Rect& rect);
void fixRedEye(Mat& img, const Rect& eyeRect);

Mat loadImage(const string& path) {
    Mat img = imread(path);
    if (img.empty()) {
        cerr << "Eroare la încărcarea imaginii: " << path << "\n";
        exit(-1);
    }
    return img;
}

Mat convertToYCrCb(const Mat& src) {
    Mat dst;
    cvtColor(src, dst, COLOR_BGR2YCrCb);
    return dst;
}

Mat segmentSkin(const Mat& ycrcb) {
    Mat mask;
    inRange(ycrcb, Scalar(0,140,100), Scalar(255,170,130), mask);
    return mask;
}

Mat cleanMask(const Mat& mask) {
    Mat tmp = mask.clone();
    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    morphologyEx(tmp, tmp, MORPH_CLOSE, k);
    morphologyEx(tmp, tmp, MORPH_OPEN, k);
    return tmp;
}

vector<vector<Point>> findSkinContours(const Mat& mask) {
    vector<vector<Point>> contours;
    Mat tmp = mask.clone();
    findContours(tmp, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}

Rect getLargestBoundingRect(const vector<vector<Point>>& contours) {
    if (contours.empty()) return Rect();
    int bestIdx = 0;
    double bestArea = contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        double a = contourArea(contours[i]);
        if (a > bestArea) {
            bestArea = a;
            bestIdx = int(i);
        }
    }
    return boundingRect(contours[bestIdx]);
}

Mat createEyeMask(const Mat& faceROI) {
    Mat gray, blurImg, blackhat, thresh, mask;
    cvtColor(faceROI, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(7,7), 1.5);
    Mat k1 = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    morphologyEx(blurImg, blackhat, MORPH_BLACKHAT, k1);
    threshold(blackhat, thresh, 10, 255, THRESH_BINARY);
    Mat k2 = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(thresh, mask, MORPH_OPEN, k2);
    morphologyEx(mask, mask, MORPH_CLOSE, k2);
    dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7,7)));
    return mask;
}

vector<Rect> detectEyes(const Mat& eyeMask) {
    auto ctrs = findSkinContours(eyeMask);
    sort(ctrs.begin(), ctrs.end(),
         [](const vector<Point>& a, const vector<Point>& b){
             return contourArea(a) > contourArea(b);
         });
    vector<Rect> eyes;
    for (size_t i = 0; i < ctrs.size() && eyes.size() < 2; ++i) {
        Rect r = boundingRect(ctrs[i]);
        if (r.width < eyeMask.cols/4 && r.width > eyeMask.cols/20 &&
            r.height < eyeMask.rows/4 && r.height > eyeMask.rows/20 &&
            r.y + r.height/2 < eyeMask.rows/2) {
            eyes.push_back(r);
        }
    }
    return eyes;
}

void drawRectangleOnImage(Mat& img, const Rect& rect) {
    if (rect.area() > 0)
        rectangle(img, rect, Scalar(0,255,0), 2);
}

void fixRedEye(Mat& img, const Rect& eyeRect) {
    Mat roi = img(eyeRect);
    const int RED_MIN        = 80;
    const int RED_OVER_GREEN = 80;
    const int RED_OVER_BLUE  = 80;

    for (int y = 0; y < roi.rows; ++y) {
        for (int x = 0; x < roi.cols; ++x) {
            Vec3b &pixel = roi.at<Vec3b>(y, x);
            int B = pixel[0], G = pixel[1], R = pixel[2];
            if (R >= RED_MIN
             && (R - G) >= RED_OVER_GREEN
             && (R - B) >= RED_OVER_BLUE) {
                pixel = Vec3b(0,0,0);
             }
        }
    }
}


int main(){
    auto start = Clock::now();
    Mat img = loadImage(R"(C:\Users\Andreea\CLionProjects\untitled4\red-eye-fix7.jpg)");
    imshow("Original", img);

    Mat ycrcb     = convertToYCrCb(img);
    Mat skinMask  = segmentSkin(ycrcb);
    Mat cleanFace = cleanMask(skinMask);
    auto faceCtrs = findSkinContours(cleanFace);
    Rect faceRect = getLargestBoundingRect(faceCtrs);
    drawRectangleOnImage(img, faceRect);
    imshow("Detected Face", img);

    Mat faceROI = img(faceRect).clone();
    Mat eyeMask = createEyeMask(faceROI);
    imshow("Eye Mask", eyeMask);
    auto eyes = detectEyes(eyeMask);

    for (auto &e : eyes) {
        Rect ge(e.x + faceRect.x,
                 e.y + faceRect.y,
                 e.width, e.height);
        drawRectangleOnImage(img, ge);
    }
    imshow("Detected Face and Eyes", img);

    for (auto &e : eyes) {
        Rect ge(e.x + faceRect.x,
                e.y + faceRect.y,
                e.width,
                e.height);
        fixRedEye(img, ge);
        drawRectangleOnImage(img, ge);
    }
    auto end = Clock::now();
    double total_ms = ms(end - start).count();

    imshow("Red-Eye Fixed", img);
    std::cout << "Total runtime: " << total_ms << " ms\n";
    waitKey();
    return 0;
}
