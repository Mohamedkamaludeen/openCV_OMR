package com.imageVisionPOC.springOpenCv;

import nu.pattern.OpenCV;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class SpringOpenCvApplication {

	static {
		System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
	}
	public static void main(String[] args) {
		SpringApplication.run(SpringOpenCvApplication.class, args);
		OpenCV.loadShared();
	}

}
