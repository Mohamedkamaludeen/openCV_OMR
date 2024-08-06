package com.imageVisionPOC.springOpenCv.Controller;

import com.imageVisionPOC.springOpenCv.Service.OmrService;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/omr")
public class OmrController {

    @Autowired
    public static OmrService Utils;

    public OmrController(OmrService utils) {
        Utils = utils;
    }

//    @PostMapping(value = "/process", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
//    public ResponseEntity<byte[]> processOmr(@RequestParam("image") MultipartFile file) throws IOException {
//        // Convert MultipartFile to Mat
//        byte[] imageBytes = file.getBytes();
//        Mat img = Imgcodecs.imdecode(new MatOfByte(imageBytes), Imgcodecs.IMREAD_COLOR);
//
//        if (img.empty()) {
//            return ResponseEntity.badRequest().body(null);
//        }
//
//        // Resize the image
//        int widthImg = 700;
//        int heightImg = 700;
//        Mat resizedImg = new Mat();
//        Size size = new Size(widthImg, heightImg);
//        Imgproc.resize(img, resizedImg, size);
//
//        // Convert Mat to BufferedImage
//        BufferedImage bufferedImage = matToBufferedImage(resizedImg);
//
//        // Convert BufferedImage to byte array
//        ByteArrayOutputStream baos = new ByteArrayOutputStream();
//        ImageIO.write(bufferedImage, "jpg", baos);
//        byte[] imageInByte = baos.toByteArray();
//
//        // Return the processed image as a response
//        return ResponseEntity.ok()
//                .contentType(MediaType.IMAGE_JPEG)
//                .body(imageInByte);
//    }
//
//    private BufferedImage matToBufferedImage(Mat mat) {
//        int type = (mat.channels() == 1) ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
//        int bufferSize = mat.channels() * mat.cols() * mat.rows();
//        byte[] b = new byte[bufferSize];
//        mat.get(0, 0, b); // get all the pixels
//        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
//        final byte[] targetPixels = ((java.awt.image.DataBufferByte) image.getRaster().getDataBuffer()).getData();
//        System.arraycopy(b, 0, targetPixels, 0, b.length);
//        return image;
//    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> uploadImage(@RequestParam("image") MultipartFile file) {
        try {
            // Convert MultipartFile to Mat
            Mat img = Utils.multipartFileToMat(file);

            // Process the image
            processImage(img);

            return ResponseEntity.ok("Image processed successfully!");
        } catch (IOException e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error processing image.");
        }
    }

    public static void processImage(Mat img) throws IOException {
        int heightImg = 700;
        int widthImg = 700;
        int questions = 5;
        int choices = 5;
        int[] ans = {1, 2, 0, 2, 4};

        Mat imgFinal;

        Imgproc.resize(img, img, new Size(widthImg, heightImg));

        imgFinal = img.clone();

        Mat imgBlank = Mat.zeros(heightImg, widthImg, CvType.CV_8UC3); // CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED

        Mat imgGray = new Mat();
        Mat imgBlur = new Mat();
        Mat imgCanny = new Mat();

        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY); // CONVERT IMAGE TO GRAY SCALE
        Imgproc.GaussianBlur(imgGray, imgBlur, new Size(5, 5), 1); // ADD GAUSSIAN BLUR
        Imgproc.Canny(imgBlur, imgCanny, 10, 70); // APPLY CANNY

        try {
            // FIND ALL CONTOURS
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(imgCanny, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            List<MatOfPoint> rectCon = Utils.rectContour(contours); // FILTER FOR RECTANGLE CONTOURS
            MatOfPoint biggestPoints = Utils.getCornerPoints(rectCon.get(0)); // GET CORNER POINTS OF THE BIGGEST RECTANGLE
            MatOfPoint gradePoints = Utils.getCornerPoints(rectCon.get(1)); // GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

            if (!biggestPoints.empty() && !gradePoints.empty()) {
                // BIGGEST RECTANGLE WARPING
                biggestPoints = Utils.reorder(biggestPoints); // REORDER FOR WARPING
                Mat pts1 = new Mat();
//                MatOfPoint2f biggestPoints2f = new MatOfPoint2f(biggestPoints.toArray());
//                biggestPoints2f.convertTo(pts1, CvType.CV_32F); // PREPARE POINTS FOR WARP
                biggestPoints.convertTo(pts1, CvType.CV_32F);

                Mat pts2 = new Mat(4, 1, CvType.CV_32FC2); // 4 rows, 1 column, 32-bit float
                pts2.put(0, 0, 0.0, 0.0);
                pts2.put(1, 0, widthImg, 0.0);
                pts2.put(2, 0, 0.0, heightImg);
                pts2.put(3, 0, widthImg, heightImg);

                Mat matrix = Imgproc.getPerspectiveTransform(pts1, pts2);
                Mat imgWarpColored = new Mat();
                Imgproc.warpPerspective(img, imgWarpColored, matrix, new Size(widthImg, heightImg));

                gradePoints = Utils.reorder(gradePoints); // REORDER FOR WARPING
                // SECOND BIGGEST RECTANGLE WARPING
                Mat ptsG1 = new Mat();
//                MatOfPoint2f gradePoints2f = new MatOfPoint2f(gradePoints.toArray());
//                gradePoints2f.convertTo(ptsG1, CvType.CV_32F); // PREPARE POINTS FOR WARP
                gradePoints.convertTo(ptsG1, CvType.CV_32F); // PREPARE POINTS FOR WARP

                Mat ptsG2 = new Mat(4, 1, CvType.CV_32FC2); // 4 rows, 1 column, 32-bit float
                ptsG2.put(0, 0, 0.0, 0.0);
                ptsG2.put(1, 0, 325, 0);
                ptsG2.put(2, 0, 0, 150);
                ptsG2.put(3, 0, 325, 150);

                Mat matrixG = Imgproc.getPerspectiveTransform(ptsG1, ptsG2);
                Mat imgGradeDisplay = new Mat();
                Imgproc.warpPerspective(img, imgGradeDisplay, matrixG, new Size(325, 150));

                // APPLY THRESHOLD
                Mat imgWarpGray = new Mat();
                Imgproc.cvtColor(imgWarpColored, imgWarpGray, Imgproc.COLOR_BGR2GRAY);
                Mat imgThresh = new Mat();
                Imgproc.threshold(imgWarpGray, imgThresh, 170, 255, Imgproc.THRESH_BINARY_INV);

                List<Mat> boxes = Utils.splitBoxes(imgThresh); // GET INDIVIDUAL BOXES
                int countR = 0;
                int countC = 0;
                int[][] myPixelVal = new int[questions][choices];  // To store the non-zero values of each box

                // Loop through each box
                for (Mat image : boxes) {
                    int totalPixels = Core.countNonZero(image);
                    myPixelVal[countR][countC] = totalPixels;
                    countC++;
                    if (countC == choices) {
                        countC = 0;
                        countR++;
                    }
                }

                // FIND THE USER ANSWERS AND PUT THEM IN A LIST
                List<Integer> myIndex = new ArrayList<>();
                for (int x = 0; x < questions; x++) {
                    int[] arr = myPixelVal[x];
                    int maxIndex = findMaxIndex(arr);
                    myIndex.add(maxIndex);
                }

                // COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
                List<Integer> grading = new ArrayList<>();
                for (int x = 0; x < questions; x++) {
                    if (ans[x] == myIndex.get(x)) {
                        grading.add(1);
                    } else {
                        grading.add(0);
                    }
                }

                // PRINT FINAL SCORE
                int totalCorrect = grading.stream().mapToInt(Integer::intValue).sum();
                double score = (double) totalCorrect / questions * 100;

                System.out.printf("Score: %d%%%n", (int) score);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new IOException("Error processing the image.", e);
        }
    }
    // Method to find the index of the maximum value in an array
    public static int findMaxIndex(int[] arr) {
        int maxIndex = 0;
        int maxValue = arr[0];

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxValue) {
                maxValue = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}


