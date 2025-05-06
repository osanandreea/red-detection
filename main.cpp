#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;



struct image_channels_bgr { Mat B, G, R; };
struct image_channels_hsv { Mat H, S, V; };
struct labels_t         { Mat labels; int no_labels; };
struct circumscribed_rectangle_coord {
    int c_min, c_max, r_min, r_max;
};


image_channels_bgr break_channels(Mat src) {
    int rows=src.rows, cols=src.cols;
    Mat B(rows,cols,CV_8UC1), G(rows,cols,CV_8UC1), R(rows,cols,CV_8UC1);
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++){
        Vec3b v = src.at<Vec3b>(i,j);
        B.at<uchar>(i,j)=v[0];
        G.at<uchar>(i,j)=v[1];
        R.at<uchar>(i,j)=v[2];
    }
    return {B,G,R};
}

image_channels_hsv bgr_2_hsv(image_channels_bgr c) {
    int rows=c.R.rows, cols=c.R.cols;
    Mat H(rows,cols,CV_32F), S(rows,cols,CV_32F), V(rows,cols,CV_32F);
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++){
        float r=c.R.at<uchar>(i,j)/255.0f,
              g=c.G.at<uchar>(i,j)/255.0f,
              b=c.B.at<uchar>(i,j)/255.0f;
        float M=max(r,max(g,b)), m=min(r,min(g,b)), C=M-m;
        V.at<float>(i,j)=M;
        S.at<float>(i,j)= (M!=0?C/M:0);
        float hh=0;
        if(C!=0){
            if(M==r) hh=60*((g-b)/C);
            else if(M==g) hh=120+60*((b-r)/C);
            else hh=240+60*((r-g)/C);
        }
        if(hh<0) hh+=360;
        H.at<float>(i,j)=hh;
    }
    return {H,S,V};
}

bool IsInside(int rows,int cols,int i,int j){
    return i>=0 && j>=0 && i<rows && j<cols;
}

const int n8_di[8]={-1,-1,-1,0,0,1,1,1};
const int n8_dj[8]={-1,0,1,-1,1,-1,0,1};

labels_t BFS_labeling(const Mat& src){
    int R=src.rows, C=src.cols, lbl=0;
    Mat L = Mat::zeros(R,C,CV_32SC1);
    queue<Point> q;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(src.at<uchar>(i,j)==0 && L.at<int>(i,j)==0){
            lbl++;
            L.at<int>(i,j)=lbl;
            q.push(Point(j,i));
            while(!q.empty()){
                Point p=q.front(); q.pop();
                for(int k=0;k<8;k++){
                    int ni=p.y+n8_di[k], nj=p.x+n8_dj[k];
                    if(IsInside(R,C,ni,nj)
                     && src.at<uchar>(ni,nj)==0
                     && L.at<int>(ni,nj)==0)
                    {
                        L.at<int>(ni,nj)=lbl;
                        q.push(Point(nj,ni));
                    }
                }
            }
        }
    }
    return {L,lbl};
}

circumscribed_rectangle_coord
compute_circumscribed_rectangle_coord(const Mat& bin){
    int R=bin.rows, C=bin.cols;
    int ymin=R, ymax=0, xmin=C, xmax=0;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(bin.at<uchar>(i,j)==0){
            ymin=min(ymin,i); ymax=max(ymax,i);
            xmin=min(xmin,j); xmax=max(xmax,j);
        }
    }
    return {xmin,xmax,ymin,ymax};
}

int main(){
    Mat img = imread(R"(C:\Users\Andreea\CLionProjects\untitled4\red-eye-fix2.jpg)");
    if(img.empty()){ cerr<<"failed to load\n"; return -1; }

    auto bgr = break_channels(img);
    auto hsv = bgr_2_hsv(bgr);

    int R=img.rows, C=img.cols;
    Mat mask(R,C,CV_8UC1);
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        float H = hsv.H.at<float>(i,j),
              S = hsv.S.at<float>(i,j),
              V = hsv.V.at<float>(i,j);
        bool isRed = (
             (H<10.0f || H>350.0f)
          && S>0.5f
          && V>0.2f
        );
        mask.at<uchar>(i,j) = isRed ? 0 : 255;
    }

    auto lab = BFS_labeling(mask);

    Mat out = img.clone();
    for(int k=1; k<=lab.no_labels; k++){
        Mat comp(R,C,CV_8UC1, Scalar(255));
        for(int i=0;i<R;i++) for(int j=0;j<C;j++){
            if(lab.labels.at<int>(i,j)==k)
                comp.at<uchar>(i,j)=0;
        }
        auto rc = compute_circumscribed_rectangle_coord(comp);
        for(int x=rc.c_min; x<=rc.c_max; x++){
            out.at<Vec3b>(rc.r_min,x) = Vec3b(0,255,0);
            out.at<Vec3b>(rc.r_max,x) = Vec3b(0,255,0);
        }
        for(int y=rc.r_min; y<=rc.r_max; y++){
            out.at<Vec3b>(y,rc.c_min) = Vec3b(0,255,0);
            out.at<Vec3b>(y,rc.c_max) = Vec3b(0,255,0);
        }
    }

    imshow("Original", img);
    imshow("Red Mask", mask);
    imshow("Detected Reds", out);
    waitKey();
    return 0;
}
// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.