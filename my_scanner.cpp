#include <iostream>
#include <opencv2/opencv.hpp>
#include<fstream>
#include<unistd.h>
using namespace std;
using namespace cv;

class Myscanner
{
private:
    Mat black_background;
    vector<Point2f> corner;
    Point2f center;

    // compute_intersect() function: compute the intersection point coordinate of two intersecting lines, given the endpoint coordinates 
    Point2f compute_intersect(Vec4i &a, Vec4i &b)
    {

        int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
        int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

        if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
        {
            cv::Point2f pt;
            pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
            pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
            return pt;
        }
        else
            return cv::Point2f(-1, -1);
    }

    // rectangle_failure() function: utilize cv::approxPolyDP() function to determine whether the points from vector<Point2f> corner
    // can be approximated to a polygen.
    bool rectangle_failure()
    {
        vector<Point2f> approx;
        approxPolyDP(Mat(corner), approx, arcLength(Mat(corner), true) * 0.02, true);
        return approx.size() != 4;
    }

    bool lines_too_close(int a, int b)
    {
        return a * a + b * b < 100;
    }

    bool point_too_close()
    {

        for (int i = 0; i < corner.size(); i++)
        {
            for (int j = i + 1; j < corner.size(); j++)
            {
                double distance = sqrt(pow((corner[i].x - corner[j].x), 2) + pow((corner[i].y - corner[j].y), 2));
                if (distance < 25)
                    return true;
            }
        }
        return false;
    }

    //sort the points from vector<Point2f> corner to match the vertexes of rectangle 
    void sort_corner_point()
    {
        Point2f upperleft, upperright, lowerleft, lowerright;

        for (auto &p : corner)
        {
            if (p.x < center.x)
            {
                if (p.y < center.y)
                    upperleft = p;
                else
                    lowerleft = p;
            }
            else
            {
                if (p.y < center.y)
                    upperright = p;
                else
                    lowerright = p;
            }
        }
        corner.clear();
        corner.push_back(upperleft);
        corner.push_back(upperright);
        corner.push_back(lowerleft);
        corner.push_back(lowerright);
    }

    // utilize filter2D() function to optimize the quality of the output picture that has been transformed
    void pic_sharpen(Mat& rect)
    {
        Mat kernel(3, 3, CV_32F, Scalar(0));
        kernel.at<float>(1, 1) = 5.0;
        kernel.at<float>(0, 1) = -1.0;
        kernel.at<float>(2, 1) = -1.0;
        kernel.at<float>(1, 0) = -1.0;
        kernel.at<float>(1, 2) = -1.0;
        filter2D(rect,rect,rect.depth(), kernel);
        imshow("transform", rect);
    }

public:
    // get the grayscale version from the original picture by function cvtColor()
    // run Canny() function to get all of the contours of the picture
    void Binary_Canny(Mat &m)
    {

        cvtColor(m, m, CV_RGB2GRAY);
        //imshow("gray",m);
        GaussianBlur(m, m, Size(5, 5), 0, 0);

        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(m, m, element);
        //imshow("dilate",m);
        Canny(m, m, 30, 120, 3);
        //imshow("canny",m);
    }

    // from all of the contours that Canny() function has discovered, select the contour with the maximum square area
    void getMaxcontour(Mat &m)
    {
        vector<vector<Point>> external_contours;
        findContours(m, external_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        int max_SquareArea = 0, index = 0;
        for (int i = 0; i < external_contours.size(); i++)
        {
            auto temp_area = fabs(contourArea(external_contours[i]));
            if (temp_area > max_SquareArea)
            {
                max_SquareArea = temp_area;
                index = i;
            }
        }
        vector<vector<Point>> final_contours;
        final_contours.push_back(external_contours[index]);
        //cout<<final_contours.size()<<endl;
        black_background = m.clone();
        black_background.setTo(0);
        drawContours(black_background, final_contours, 0, Scalar(255), 1);
        imshow("contour", black_background);
    }

    //utilize HoughLinesP() function to get the approximate 4 edges of the contour
    void get_line(Mat &m)
    {
        vector<Vec4i> lines;

        int para = 10, round = 1;
        for (; para < 250; para++, round++)
        {
            lines.clear();

            HoughLinesP(black_background, lines, 1, CV_PI / 180, para, 30, 10);

            set<int> ErasePointSet;
            for (int i = 0; i < lines.size(); i++)
            {
                for (int j = i + 1; j < lines.size(); j++)
                {
                    if (lines_too_close(abs(lines[i][0] - lines[j][0]),
                                        abs(lines[i][1] - lines[j][1])) &&
                        (lines_too_close(abs(lines[i][2] - lines[j][2]),
                                         abs(lines[i][3] - lines[j][3]))))
                    {
                        ErasePointSet.insert(j);
                    }
                }
            }

            for (int i = lines.size(); i > 0; i--)
            {
                if (ErasePointSet.find(i) != ErasePointSet.end())
                {
                    lines.erase(lines.begin() + i - 1);
                }
            }
            cout << "round " << round << " "
                 << "line size " << lines.size() << endl;

            if (lines.size() != 4)
                continue;

            for (int i = 0; i < lines.size(); i++)
            {
                for (int j = i + 1; j < lines.size(); j++)
                {
                    Point2f pt = compute_intersect(lines[i], lines[j]);
                    if (pt.x >= 0 && pt.x <= m.cols && pt.y >= 0 && pt.y <= m.rows)
                    {
                        corner.push_back(pt);
                    }
                }
            }

            //cout<<corner.size()<<endl;
            if (corner.size() != 4 || rectangle_failure() || point_too_close())
                continue;
            else
            {
                cout << "we find it." << endl;
                break;
            }
        }
    }

    //determine the 4 intersection points from the approximate edges calculated by HoughLinesP() function 
    void draw_intersectpoint(Mat &m)
    {

        //求出中心
        for (int i = 0; i < corner.size(); i++)
            center += corner[i];
        center *= (1. / corner.size());

        //画出交点
        cv::circle(m, corner[0], 3, CV_RGB(255, 0, 0), -1);
        cv::circle(m, corner[1], 3, CV_RGB(0, 255, 0), -1);
        cv::circle(m, corner[2], 3, CV_RGB(0, 0, 255), -1);
        cv::circle(m, corner[3], 3, CV_RGB(255, 255, 255), -1);
        cv::circle(m, center, 3, CV_RGB(0, 0, 0), -1);
        imshow("intersect points", m);
    }

    //determine the height & width of the transformed picture in advance 
    void determine_height_width(double &dst_height, double &dst_width)
    {

        sort_corner_point();
        double h1, h2, w1, w2;

        w1 = sqrt(pow((corner[1].x - corner[0].x), 2) + pow((corner[1].y - corner[0].y), 2));
        w2 = sqrt(pow((corner[3].x - corner[2].x), 2) + pow((corner[3].y - corner[2].y), 2));
        h1 = sqrt(pow((corner[2].x - corner[0].x), 2) + pow((corner[2].y - corner[0].y), 2));
        h2 = sqrt(pow((corner[3].x - corner[1].x), 2) + pow((corner[3].y - corner[1].y), 2));

        //cout<<h1<<" "<<h2<<" "<<w1<<" "<<w2<<endl;

        dst_height = 1.25 * max(h1, h2);
        dst_width = 1.25 * max(w1, w2);
    }

    //get the transformed picture (work as a scanner app) by warpPerspective() and getPerspectiveTransform() 
    void pic_transform(Mat &src, double &dst_height, double &dst_width)
    {

        Mat rect = cv::Mat::zeros(dst_height, dst_width, CV_8UC3);
        vector<Point2f> rect_points;
        rect_points.push_back(Point2f(0, 0));
        rect_points.push_back(Point2f(rect.cols, 0));
        rect_points.push_back(Point2f(0, rect.rows));
        rect_points.push_back(Point2f(rect.cols, rect.rows));
        cout << rect_points.size() << " " << corner.size() << endl;
        Mat transform = getPerspectiveTransform(corner, rect_points);
        warpPerspective(src, rect, transform, rect.size());

        pic_sharpen(rect);
    }
};

void operation(string filename)
{

    Myscanner scanner;
    Mat src = imread(filename);

    Mat img_1 = src.clone();
    scanner.Binary_Canny(img_1);
    scanner.getMaxcontour(img_1);

    Mat img_2 = img_1.clone();
    scanner.get_line(img_2);
    Mat img_3 = src.clone();
    scanner.draw_intersectpoint(src);
    double height = 0, width = 0;
    scanner.determine_height_width(height, width);
    scanner.pic_transform(img_3, height, width);

    waitKey(0);
    destroyAllWindows();
}

int main()
{   

    //read the file from "test_img/img_name.txt" to test the pictures in order 
    ifstream ifs;
    ifs.open("test_img/img_name.txt");
    if(!ifs.is_open()) {
        cout<<"file failure."<<endl;
        return 0;
    }

    string filename_plus;
    while(ifs>>filename_plus){
        string file="test_img/";
        file+=filename_plus;
        cout<<file<<endl;
        operation(file);
    }

    ifs.close();
    return 0;
}