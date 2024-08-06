package com.imageVisionPOC.springOpenCv.Service;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfPoint;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.*;

import java.util.ArrayList;
import java.util.List;
@Service
public class OmrService {
    // Updated method to include labels parameter
//    public static Mat stackImages(Mat[][] imgArray, double scale) {
//        int rows = imgArray.length;
//        int cols = imgArray[0].length;
//        int width = imgArray[0][0].cols();
//        int height = imgArray[0][0].rows();
//
//        Mat imageBlank = Mat.zeros(height, width, CvType.CV_8UC3);
//        Mat[] hor = new Mat[rows];
//        Mat[] horCon = new Mat[rows];
//
//        for (int x = 0; x < rows; x++) {
//            Mat[] row = imgArray[x];
//            Mat[] rowResized = new Mat[cols];
//            for (int y = 0; y < cols; y++) {
//                rowResized[y] = new Mat();
//                Imgproc.resize(row[y], rowResized[y], new Size(0, 0), scale, scale, Imgproc.INTER_LINEAR);
//                if (rowResized[y].channels() == 1) {
//                    Imgproc.cvtColor(rowResized[y], rowResized[y], Imgproc.COLOR_GRAY2BGR);
//                }
//            }
//            hor[x] = new Mat();
//            Core.hconcat(Arrays.asList(rowResized), hor[x]);
//            horCon[x] = new Mat();
//            Core.hconcat(Arrays.asList(rowResized), horCon[x]);
//        }
//
//        Mat ver = new Mat();
//        Mat verCon = new Mat();
//        if (labels.length > 0) {
//            // Create a temporary image for labels
//            Mat temp = new Mat();
//            Core.vconcat(Arrays.asList(hor), temp);
//            ver = temp.clone();
//            for (int d = 0; d < rows; d++) {
//                for (int c = 0; c < cols; c++) {
//                    // Add label background
//                    Imgproc.rectangle(ver, new Point(c * width, d * height),
//                            new Point(c * width + labels[d][c].length() * 13 + 27, 30 + d * height),
//                            new Scalar(255, 255, 255), Imgproc.FILLED);
//                    // Add label text
//                    Imgproc.putText(ver, labels[d][c], new Point(c * width + 10, d * height + 20),
//                            Imgproc.FONT_HERSHEY_COMPLEX, 0.7, new Scalar(255, 0, 255), 2);
//                }
//            }
//        } else {
//            ver = new Mat();
//            Core.vconcat(Arrays.asList(hor), ver);
//        }
//
//        verCon = new Mat();
//        Core.vconcat(Arrays.asList(horCon), verCon);
//
//        return ver;
//    }

    public static MatOfPoint reorder(MatOfPoint points) {
        // Reshape the MatOfPoint to a 2D Mat with 4 rows and 2 columns
        Mat pointsMat = points.reshape(1, 4);

        // Create a new Mat to store the reordered points with the same type and 4 rows and 1 column
        Mat pointsNewMat = Mat.zeros(4, 1, CvType.CV_32FC2);

        // Create a Mat to store the result of the sum with 1 column
        Mat add = new Mat(pointsMat.rows(), 1, CvType.CV_32FC1);

        // Sum across columns (axis 1)
        for (int i = 0; i < pointsMat.rows(); i++) {
            Mat row = pointsMat.row(i);
            Scalar sum = Core.sumElems(row);
            add.put(i, 0, sum.val[0]);
        }

        // Find the index of the minimum and maximum values in 'add'
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(add);
        int minIndex = (int) minMaxLocResult.minLoc.y;
        int maxIndex = (int) minMaxLocResult.maxLoc.y;

        // Extract rows based on min and max index
        Mat row1 = pointsMat.row(minIndex);
        row1.copyTo(pointsNewMat.row(0));

        Mat row2 = pointsMat.row(maxIndex);
        row2.copyTo(pointsNewMat.row(3));

        // Compute the difference along axis 1 (columns) with correct type
        Mat diff = new Mat(pointsMat.rows(), pointsMat.cols() - 1, CvType.CV_32FC1);
        for (int i = 0; i < pointsMat.rows(); i++) {
            double[] current = pointsMat.get(i, 0);
            for (int j = 0; j < pointsMat.cols() - 1; j++) {
                double[] next = pointsMat.get(i, j + 1);
                if (current != null && next != null) {
                    double[] result = new double[next.length];
                    for (int k = 0; k < next.length; k++) {
                        result[k] = next[k] - current[k];
                    }
                    diff.put(i, j, result);
                }
            }
        }

        // Find the index of the minimum and maximum values in 'diff'
        Core.MinMaxLocResult minMaxLocResultDiff = Core.minMaxLoc(diff);
        int minIndex2 = (int) minMaxLocResultDiff.minLoc.y;
        int maxIndex2 = (int) minMaxLocResultDiff.maxLoc.y;

        // Extract rows based on min and max index in 'diff'
        Mat row3 = pointsMat.row(minIndex2);
        row3.copyTo(pointsNewMat.row(1));

        Mat row4 = pointsMat.row(maxIndex2);
        row4.copyTo(pointsNewMat.row(2));

        // Convert the Mat to List<Point>
        List<Point> pointsList = new ArrayList<>();
        for (int i = 0; i < pointsNewMat.rows(); i++) {
            double[] coords = pointsNewMat.get(i, 0);
            pointsList.add(new Point(coords[0], coords[1]));
        }

        // Convert List<Point> to MatOfPoint
        MatOfPoint pointsNew = new MatOfPoint();
        pointsNew.fromList(pointsList);

        return pointsNew;
    }

    public  List<MatOfPoint> rectContour(List<MatOfPoint> contours) {
        List<MatOfPoint> rectCon = new ArrayList<>();
        double maxArea = 0;
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > 50) {
                double peri = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approx, 0.02 * peri, true);
                if (approx.toArray().length == 4) {
                    rectCon.add(contour);
                }
            }
        }
        rectCon.sort((a, b) -> Double.compare(Imgproc.contourArea(b), Imgproc.contourArea(a)));
        return rectCon;
    }

//    public static Mat getCornerPoints(Mat contour) {
//        MatOfPoint2f approx = new MatOfPoint2f();
//        double peri = Imgproc.arcLength(new MatOfPoint2f(contour), true); // LENGTH OF CONTOUR
//        Imgproc.approxPolyDP(new MatOfPoint2f(contour), approx, 0.02 * peri, true); // APPROXIMATE THE POLY TO GET CORNER POINTS
//        return approx;
//    }
public static MatOfPoint getCornerPoints(MatOfPoint contour) {
    // Convert MatOfPoint to MatOfPoint2f
    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

    // Calculate perimeter of the contour
    double peri = Imgproc.arcLength(contour2f, true);

    // Approximate the contour to get corner points
    MatOfPoint2f approxCurve = new MatOfPoint2f();
    Imgproc.approxPolyDP(contour2f, approxCurve, 0.02 * peri, true);

    // Convert the result back to MatOfPoint
    return new MatOfPoint(approxCurve.toArray());
}

    public static List<Mat> splitBoxes(Mat img) {
        // Determine the number of rows and columns to split into
        int numRows = 5;
        int numCols = 5;

        // Get the dimensions of the image
        int imgHeight = img.rows();
        int imgWidth = img.cols();

        // Calculate the height and width of each slice
        int rowHeight = imgHeight / numRows;
        int colWidth = imgWidth / numCols;

        // Create a list to hold the boxes
        List<Mat> boxes = new ArrayList<>();

        // Split the image into rows
        for (int i = 0; i < numRows; i++) {
            int yStart = i * rowHeight;
            int yEnd = (i == numRows - 1) ? imgHeight : (i + 1) * rowHeight;
            Rect rowRect = new Rect(0, yStart, imgWidth, yEnd - yStart);
            Mat row = new Mat(img, rowRect);

            // Split each row into columns
            for (int j = 0; j < numCols; j++) {
                int xStart = j * colWidth;
                int xEnd = (j == numCols - 1) ? imgWidth : (j + 1) * colWidth;
                Rect colRect = new Rect(xStart, 0, xEnd - xStart, row.rows());
                Mat box = new Mat(row, colRect);
                boxes.add(box);
            }
        }
        return boxes;
    }

//    public static Mat drawGrid(Mat img) {
//        int questions=5;
//        int choices=5;
//        int secW = img.cols() / questions;
//        int secH = img.rows() / choices;
//        for (int i = 0; i <= questions; i++) {
//            Imgproc.line(img, new Point(0, secH * i), new Point(img.cols(), secH * i), new Scalar(255, 255, 0), 2);
//        }
//        for (int i = 0; i <= choices; i++) {
//            Imgproc.line(img, new Point(secW * i, 0), new Point(secW * i, img.rows()), new Scalar(255, 255, 0), 2);
//        }
//        return img;
//    }

//    public static Mat showAnswers(Mat img, int[] myIndex, int[] grading, int[] ans) {
//        int questions=5;
//        int choices=5;
//        int secW = img.cols() / questions;
//        int secH = img.rows() / choices;
//        for (int x = 0; x < questions; x++) {
//            int myAns = myIndex[x];
//            int cX = (myAns * secW) + secW / 2;
//            int cY = (x * secH) + secH / 2;
//            Scalar color;
//            if (grading[x] == 1) {
//                color = new Scalar(0, 255, 0);
//                Imgproc.circle(img, new Point(cX, cY), 50, color, Imgproc.FILLED);
//            } else {
//                color = new Scalar(0, 0, 255);
//                Imgproc.circle(img, new Point(cX, cY), 50, color, Imgproc.FILLED);
//                color = new Scalar(0, 255, 0);
//                int correctAns = ans[x];
//                Imgproc.circle(img, new Point((correctAns * secW) + secW / 2, (x * secH) + secH / 2), 20, color, Imgproc.FILLED);
//            }
//        }
//        return img;
//    }

    public Mat multipartFileToMat(MultipartFile file) throws IOException {
//        BufferedImage bufferedImage = ImageIO.read(new ByteArrayInputStream(file.getBytes()));
//        Mat mat = bufferedImageToMat(bufferedImage);
        // Convert MultipartFile to Mat
        byte[] imageBytes = file.getBytes();
        Mat img = Imgcodecs.imdecode(new MatOfByte(imageBytes), Imgcodecs.IMREAD_COLOR);
        return img;
    }
}
