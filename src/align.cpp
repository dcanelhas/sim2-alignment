#include "Eigen/Core"
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <limits>
#include <vector>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

template <class T>
T Interpolate(cv::Mat &image, float y,float x)
{
  float xd, yd;  
  float k1 = modff(x,&xd);
  float k2 = modff(y,&yd);
  int xi = int(xd);
  int yi = int(yd);

  int f1 = xi < image.rows-1;  // Check that pixels to the right  
  int f2 = yi < image.cols-1; // and to down direction exist.

  T px1 = image.at<T>(yi  , xi);
  T px2 = image.at<T>(yi  , xi+1);
  T px3 = image.at<T>(yi+1, xi);
  T px4 = image.at<T>(yi+1, xi+1);      
  
  // Interpolate pixel intensity.
  T interpolated_value = 
  (1.0-k1)*(1.0-k2)*px1 +
  (f1 ? ( k1*(1.0-k2)*px2 ):0) +
  (f2 ? ( (1.0-k1)*k2*px3 ):0) +            
  ((f1 && f2) ? ( k1*k2*px4 ):0);

  return interpolated_value;
}

void Align(cv::Mat &source, cv::Mat &target, int max_iterations, Eigen::Matrix3f &transformation)
{

  // Find the 2-D similarity transform that best aligns the two images (uniform scale, rotation and translation)
  const float EPS = 1e-8; // Threshold value for termination criteria.
  const float HUBER_LOSS = 0.60;
  cv::Mat debug;
  
  cv::Mat source_gradient_row;    // Gradient of I in X direction.
  cv::Mat source_gradient_col;    // Gradient of I in Y direction.
  cv::Mat steepest_descent;       // Steepest descent images.

  // Here we will store matrices.
  Eigen::Matrix3f W;        // Current value of warp W(x,p)
  Eigen::Matrix3f dW;       // Warp update.
  Eigen::Vector3f X;        // Point in coordinate frame of source.
  Eigen::Vector3f Z;        // Point in coordinate frame of target.

  Eigen::Matrix4f H;        // Approximate Hessian.
  Eigen::Vector4f b;        // Vector in the right side of the system of linear equations.
  Eigen::Vector4f delta_p;  // Parameter update value.
  
  // Create images.
  source_gradient_row = cv::Mat(source.rows, source.cols, CV_32FC1);
  source_gradient_col = cv::Mat(source.rows, source.cols, CV_32FC1);
  steepest_descent =    cv::Mat(source.rows, source.cols, CV_32FC4);

  //The "magic number" appearing at the end in the following is simply the inverse 
  //of the absolute sum of the weights in the matrix representing the Scharr filter.
  cv::Scharr(source, source_gradient_row, -1, 0, 1, 1.0/32.0); 
  cv::Scharr(source, source_gradient_col, -1, 1, 0, 1.0/32.0);   
  
  H = Eigen::Matrix4f::Zero();
  float h00 = 0.0, h01 = 0.0, h02 = 0.0, h03 = 0.0; 
  float h10 = 0.0, h11 = 0.0, h12 = 0.0, h13 = 0.0; 
  float h20 = 0.0, h21 = 0.0, h22 = 0.0, h23 = 0.0; 
  float h30 = 0.0, h31 = 0.0, h32 = 0.0, h33 = 0.0;

  #pragma omp parallel for \
  reduction(+:h00,h01,h02,h03,h10,h11,h12,h13,h20,h21,h22,h23,h30,h31,h32,h33)
  for(int row=0; row<source.rows; row++)//
  {
    #pragma unroll
    for(int col=0; col<source.cols; col++) //
    {
      // Evaluate image gradient
      Eigen::Matrix<float,1,2> image_jacobian;
      image_jacobian << source_gradient_row.at<float>(row,col),
                        source_gradient_col.at<float>(row,col);

      // printf("image jacobian = %f, %f", image_jacobian(0),image_jacobian(1));


      Eigen::Matrix<float,2,4> warp_jacobian;
      warp_jacobian <<  1, 0 , row,  row,
                        0, 1 , -col, col;

      Eigen::Vector4f Jacobian = (image_jacobian*warp_jacobian).transpose();                          

      for(int dim = 0; dim<4; ++dim)
      steepest_descent.at<cv::Vec4f>(row, col)[dim] = Jacobian(dim);

      Eigen::Matrix4f Hpart = Jacobian*Jacobian.transpose();

      h00+=Hpart(0,0); h01+=Hpart(0,1); h02+=Hpart(0,2); h03+=Hpart(0,3); 
      h10+=Hpart(1,0); h11+=Hpart(1,1); h12+=Hpart(1,2); h13+=Hpart(1,3); 
      h20+=Hpart(2,0); h21+=Hpart(2,1); h22+=Hpart(2,2); h23+=Hpart(2,3); 
      h30+=Hpart(3,0); h31+=Hpart(3,1); h32+=Hpart(3,2); h33+=Hpart(3,3); 
    }
  }

  H <<
  h00,h01,h02,h03,
  h10,h11,h12,h13,
  h20,h21,h22,h23,
  h30,h31,h32,h33;

  W = transformation;

  // Iterate
  int iter=0; // number of current iteration
  while(iter < max_iterations)
  {
    target.copyTo(debug);
    iter++; // Increment iteration counter

    uint pixel_count = 0; // Count of processed pixels
    
    float b0=0.0, b1=0.0, b2=0.0, b3=0.0;
    float mean_error = 0.0;
        
    #pragma omp parallel for \
    reduction(+:mean_error, pixel_count, b0, b1, b2, b3)
    for(int row=0; row<source.rows; row++)
    {
      #pragma unroll
      for(int col=0; col<source.cols; col++)
      {
        // Set vector X with pixel coordinates (u,v,1)
        X = Eigen::Vector3f(row, col, 1.0);
        Z = W*X;
        
        float row2 = Z(0);
        float col2 = Z(1);

        // Get the nearest integer pixel coords (u2i;v2i).
        int row2i = int(floor(row2));
        int col2i = int(floor(col2));

        if(row2i>=0 && row2i<target.rows && // check if pixel is inside I.
          col2i>=0 && col2i<target.cols)
        {
          pixel_count++;

          // Calculate intensity of a transformed pixel with sub-pixel accuracy
          // using bilinear interpolation.
          float I2 = Interpolate<float>(target, row2, col2);
          
          debug.at<float>(row2i,col2i) = source.at<float>(row,col);

          // Calculate image difference D = I(W(x,p))-T(x).
          float D = I2 - target.at<float>(row, col);

          // Update mean error value.
          mean_error += fabsf(D);

          // Add a term to b matrix.

          Eigen::Vector4f db;
          db << steepest_descent.at<cv::Vec4f>(row, col)[0],
                steepest_descent.at<cv::Vec4f>(row, col)[1],
                steepest_descent.at<cv::Vec4f>(row, col)[2],
                steepest_descent.at<cv::Vec4f>(row, col)[3];
         
          db *= (fabsf(D) < HUBER_LOSS) ? D : D*HUBER_LOSS/fabsf(D);

          if(!std::isnan(db.dot(db))) 
          {
            b0 += db(0); 
            b1 += db(1); 
            b2 += db(2); 
            b3 += db(3); 

          }
        } 
      }
    }
    // std::cout<< "residual:" << mean_error/pixel_count << std::endl; 

    cv::imshow("Initial Alignment", debug);
    cv::waitKey(2);
    
    b = Eigen::Vector4f(b0,b1,b2,b3);

    // std::cout << "b:" << std::endl;
    // std::cout << b << std::endl;
    // // Find parameter increment. 
    
    H.ldlt().solve(b);
    Eigen::Matrix2f skew;
    skew << 0.0, -1.0, 
            1.0,  0.0;

    // Rodrigues' formula:
    Eigen::Matrix2f R = Eigen::Matrix2f::Identity() + sinf(delta_p(2))*skew +  (1-cosf(delta_p(2)))*(skew*skew.transpose() - Eigen::Matrix2f::Identity());
    
    Eigen::Matrix3f T;
    T << R(0,0)+delta_p(3), R(0,1),             delta_p(0),
         R(1,0),            R(1,1)+delta_p(3),  delta_p(1),
            0.0,                          0.0,          1;
 
    dW = T.inverse()*W;
    W = dW;

    // Check termination critera.
    if(delta_p.norm()<=EPS)
      {      
        transformation = W;
        std::cout << "Terminated in " << iter << " iterations." <<std::endl;
        cv::imshow("Final Alignment", debug);
        cv::waitKey(500);
        return;
      }
  } // iteration
  std::cout << "Maximum iterations reached (" << iter << ")." <<std::endl;
  transformation = W;
  return;
}//function

int main(int argc, char **argv)
{
  cv::Mat img_src_c = cv::imread(argv[1], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR ); // Read the file
  cv::Mat img_src_f; img_src_c.convertTo(img_src_f, CV_32F);
  cv::Mat img_trg_c = cv::imread(argv[2], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR ); // Read the OHTER file
  cv::Mat img_trg_f; img_trg_c.convertTo(img_trg_f, CV_32F);

  img_src_f/=255;
  img_trg_f/=255;

  cv::Mat img_src_blur;
  cv::Mat img_trg_blur;

  float sigma = 0.95; //computed from filter size n=3 in [sigma = 0.3(n/2 - 1) + 0.8]
  cv::GaussianBlur(img_src_f, img_src_blur, cv::Size(3,3), sigma);
  cv::GaussianBlur(img_trg_f, img_trg_blur, cv::Size(3,3), sigma);
  
  cv::Mat img_src_half;
  cv::Mat img_trg_half;
  cv::resize(img_src_blur, img_src_half, cv::Size(0,0), 0.5, 0.5);
  cv::resize(img_trg_blur, img_trg_half, cv::Size(0,0), 0.5, 0.5);
  
  cv::GaussianBlur(img_src_half, img_src_blur, cv::Size(3,3), sigma);
  cv::GaussianBlur(img_trg_half, img_trg_blur, cv::Size(3,3), sigma);
  
  cv::Mat img_src_quarter;
  cv::Mat img_trg_quarter;
  cv::resize(img_src_blur, img_src_quarter, cv::Size(0,0), 0.5, 0.5);
  cv::resize(img_trg_blur, img_trg_quarter, cv::Size(0,0), 0.5, 0.5);
  

  Eigen::Matrix3f initial_guess;

  initial_guess <<
  1.4, -0.0,  210/4,
  0.0,  1.4,  110/4,
  0,      0,    1;

  Align(img_src_quarter, img_trg_quarter, 300, initial_guess);

  std::cout << "W:" << std::endl;
  std::cout << initial_guess << std::endl;

  initial_guess(0,2) *= 2;
  initial_guess(1,2) *= 2;
  Align(img_src_half, img_trg_half, 200, initial_guess);
  initial_guess(0,2) *= 2;
  initial_guess(1,2) *= 2;
  Align(img_src_f, img_trg_f, 100, initial_guess);



  return 0;
}
