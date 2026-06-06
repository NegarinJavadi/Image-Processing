A small computer vision project that goes through the classic image processing techniques — noise, filtering, edge detection, thresholding, segmentation — and then uses them to build a machine learning.
The first part is a visual walkthrough. All the filters applied to the same image, or the different edge detectors.
Then add Gaussian noise at random levels (300 times by default), and for each noisy version it:

- applies three filters (Gaussian, median, bilateral) and measures how close each
  result is to the original using PSNR and SSIM
- records which filter scored best — this becomes the label
- pulls out a few extra numbers from the image: edge density, gradient strength,
  Laplacian variance, the Otsu threshold level, and the foreground ratio

That gives a small dataset. Then it trains three models (Random Forest, SVM,
Gradient Boosting) to predict the best filter from those features, compares them
with cross-validation, and plots the feature importances for the best one.

The settings live at the top of `main()`, so just open the file and change them:

python
SECTION   = "1"     # 1=noise  2=filters  3=edges  4=threshold  5=segmentation
RUN_EVAL  = True    # print MAE / MSE / RMSE / PSNR / SSIM for the noise types
RUN_ML    = True    # build the dataset and train the models
N_SAMPLES = 300     # how many noisy samples to generate


