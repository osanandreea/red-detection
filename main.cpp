#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// ——————————————————————————————————————————————————————————————
// Prot­otipuri funcții
// ——————————————————————————————————————————————————————————————
Mat           loadImage(const string& path);
Mat           convertToYCrCb(const Mat& src);
Mat           segmentSkin(const Mat& ycrcb);
Mat           cleanMask(const Mat& mask);
vector<vector<Point>> findSkinContours(const Mat& mask);
Rect          getLargestBoundingRect(const vector<vector<Point>>& contours);
Mat           createEyeMask(const Mat& faceROI);
vector<Rect>  detectEyes(const Mat& eyeMask);
void          drawRectangleOnImage(Mat& img, const Rect& rect);

// ——————————————————————————————————————————————————————————————
// Implementări
// ——————————————————————————————————————————————————————————————

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
    inRange(ycrcb,
            Scalar(  0, 140, 100),
            Scalar(255, 170, 130),
            mask);
    return mask;
}

Mat cleanMask(const Mat& mask) {
    Mat tmp = mask.clone();
    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    morphologyEx(tmp, tmp, MORPH_CLOSE, k);
    morphologyEx(tmp, tmp, MORPH_OPEN,  k);
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

// Construiește o mască CV_8UC1 a zonei ochilor (umbre intunecate)
Mat createEyeMask(const Mat& faceROI) {
    Mat gray, blurImg, blackhat, thresh, mask;
    // 1) Gri și blur
    cvtColor(faceROI, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(7,7), 1.5);

    // 2) Black-hat ca să evidențiem umbrele din iris
    Mat k1 = getStructuringElement(MORPH_ELLIPSE, Size(15,15));
    morphologyEx(blurImg, blackhat, MORPH_BLACKHAT, k1);

    // 3) Threshold: umbrele (zonele întunecate) devin 255
    threshold(blackhat, thresh, 10, 255, THRESH_BINARY);

    // 4) Curățire și mică dilatare
    Mat k2 = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(thresh, mask, MORPH_OPEN,   k2);
    morphologyEx(mask, mask, MORPH_CLOSE,    k2);
    dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7,7)));

    return mask;
}

// Din masca CV_8UC1 a ochilor extrage cele două rect-uri
vector<Rect> detectEyes(const Mat& eyeMask) {
    auto ctrs = findSkinContours(eyeMask);
    // sort descrescător după arie
    sort(ctrs.begin(), ctrs.end(),
         [](const vector<Point>& a, const vector<Point>& b){
             return contourArea(a) > contourArea(b);
         });
    vector<Rect> eyes;
    for (size_t i = 0; i < ctrs.size() && eyes.size() < 2; ++i) {
        Rect r = boundingRect(ctrs[i]);
        // filtrăm dupa dimensiune și poziție (sus-față)
        if (r.width  < eyeMask.cols/4  && r.width  > eyeMask.cols/20 &&
            r.height < eyeMask.rows/4  && r.height > eyeMask.rows/20 &&
            r.y + r.height/2 < eyeMask.rows/2)
        {
            eyes.push_back(r);
        }
    }
    return eyes;
}

void drawRectangleOnImage(Mat& img, const Rect& rect) {
    if (rect.area() > 0)
        rectangle(img, rect, Scalar(0,255,0), 2);
}

// ——————————————————————————————————————————————————————————————
// main
// ——————————————————————————————————————————————————————————————
int main(){
    // 1) Încarcă și afișează originalul
    Mat img       = loadImage(R"(C:\Users\Andreea\CLionProjects\untitled4\red-eye-fix2.jpg)");
    imshow("Original", img);

    // 2) Detectează și desenează fața
    Mat ycrcb     = convertToYCrCb(img);
    Mat skinMask  = segmentSkin(ycrcb);
    Mat cleanFace = cleanMask(skinMask);
    auto faceCtrs = findSkinContours(cleanFace);
    Rect faceRect= getLargestBoundingRect(faceCtrs);
    drawRectangleOnImage(img, faceRect);
    imshow("Detected Face", img);

    // 3) Creează mască ochi și detectează ochii
    Mat faceROI = img(faceRect).clone();
    Mat eyeMask = createEyeMask(faceROI);
    imshow("Eye Mask", eyeMask);
    auto eyes = detectEyes(eyeMask);

    // 4) Desenează ochii pe imaginea globală
    for (auto &e : eyes) {
        Rect ge(e.x + faceRect.x, e.y + faceRect.y, e.width, e.height);
        drawRectangleOnImage(img, ge);
    }
    imshow("Detected Face and Eyes", img);

    waitKey();
    return 0;
}
