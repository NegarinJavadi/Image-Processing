import logging 
from dataclasses import dataclass 
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# 1. Data classes
@dataclass
class EvalResult:
    name: str
    mae: float
    mse: float
    rmse: float
    psnr: float
    ssim: float

    def __str__(self) -> str:
        return (
            f"\n{'='*40}\n{self.name}\n"
            f"  MAE  = {self.mae:.2f}\n"
            f"  MSE  = {self.mse:.2f}\n"
            f"  RMSE = {self.rmse:.2f}\n"
            f"  PSNR = {self.psnr:.2f} dB\n"
            f"  SSIM = {self.ssim:.4f}"
        )

@dataclass
class EdgeFeatures:
    canny_density: float      
    sobel_mean: float        
    laplacian_var: float   

@dataclass
class ThresholdFeatures:
    otsu_level: float       
    foreground_ratio: float   

# 2. Noise:
class NoiseGenerator:

    @staticmethod 
    def gaussian(image: np.ndarray, std: float = 25.0) -> np.ndarray:
        # low std : low noise . high std : high noise 
        """Additive Gaussian noise with zero mean."""
        noise = np.random.normal(0, std, image.shape)
        # mean = 0 , std: shedate noise
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def salt_and_pepper(image: np.ndarray, prob: float = 0.03) -> np.ndarray:
        # low prob, low noise
        # high prob, tasvir az beyn mire
        noisy = image.copy()
        mask = np.random.random(image.shape) < prob
        salt = np.random.random(image.shape) < 0.5
        noisy[mask & salt] = 255
        noisy[mask & ~salt] = 0
        return noisy

    @staticmethod
    def impulse(image: np.ndarray, prob: float = 0.05) -> np.ndarray:
        noisy = image.copy()
        mask = np.random.random(image.shape) < prob
        noisy[mask] = np.random.randint(0, 256, image.shape, dtype=np.uint8)[mask]
        return noisy

# 3. Filters:
class ImageFilter:

    @staticmethod
    def gaussian(image: np.ndarray, ksize: int = 7) -> np.ndarray:
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    @staticmethod
    def average(image: np.ndarray, ksize: int = 7) -> np.ndarray:
        return cv2.blur(image, (ksize, ksize))

    @staticmethod
    def box(image: np.ndarray, ksize: int = 7) -> np.ndarray:
        return cv2.boxFilter(image, -1, (ksize, ksize))

    @staticmethod
    def median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def bilateral(image: np.ndarray, d: int = 9,sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def non_local_means(image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image)

    @staticmethod
    def wiener(image: np.ndarray) -> np.ndarray:
        result = wiener(image.astype(np.float32))
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def edge_preserving(image: np.ndarray) -> np.ndarray:
        return cv2.edgePreservingFilter(image)

# 4. Edge detection 
class EdgeDetector:

    @staticmethod
    def sobel(image: np.ndarray) -> np.ndarray:
        sx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        return np.uint8(np.abs(sx))

    @staticmethod
    def prewitt(image: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def canny(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
        return cv2.Canny(image, low, high)

    @staticmethod
    def laplacian(image: np.ndarray) -> np.ndarray:
        result = cv2.Laplacian(image, cv2.CV_64F)
        return np.uint8(np.abs(result))

    @classmethod
    def extract_features(cls, image: np.ndarray) -> EdgeFeatures:
        canny_edges = cls.canny(image)
        canny_density = float(np.count_nonzero(canny_edges)) / canny_edges.size

        sobel_map = cls.sobel(image).astype(np.float32)
        sobel_mean = float(sobel_map.mean())

        lap = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_var = float(lap.var())

        return EdgeFeatures(
            canny_density=canny_density,
            sobel_mean=sobel_mean,
            laplacian_var=laplacian_var)

# 5. Thresholding 
class Threshold:

    @staticmethod
    def binary(image: np.ndarray, threshold: int = 127) -> np.ndarray:
        _, th = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return th

    @staticmethod
    def adaptive(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, block_size, C)

    @staticmethod
    def otsu(image: np.ndarray) -> tuple[np.ndarray, float]:
        val, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th, float(val) / 255.0

    @classmethod
    def extract_features(cls, image: np.ndarray) -> ThresholdFeatures:

        """
        image: Grayscale uint8 image.
        0: pas zamine
        255: pish zamine
        """
        th, otsu_level = cls.otsu(image)
        foreground_ratio = float(np.count_nonzero(th)) / th.size
        return ThresholdFeatures(otsu_level=otsu_level, foreground_ratio=foreground_ratio)

# 6. Segmentation
class Segmentation:
    @staticmethod
    def morphology(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def kmeans(image: np.ndarray, k: int = 4, iterations: int = 10) -> np.ndarray:
        Z = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
        _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        return center[label.flatten()].reshape(image.shape)

# 7. Evaluation
def evaluate(original: np.ndarray, processed: np.ndarray, name: str = "") -> EvalResult:

    if original.shape != processed.shape:
        raise ValueError(f"Shape mismatch: original={original.shape}, processed={processed.shape}")
    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)
    diff = np.abs(orig_f - proc_f)

    return EvalResult(
        name=name,
        mae=float(diff.mean()),
        mse=float((diff ** 2).mean()),
        rmse=float(np.sqrt((diff ** 2).mean())),
        psnr=float(peak_signal_noise_ratio(original, processed)),
        ssim=float(structural_similarity(original, processed)))

# 8. Feature engineering 
def extract_sample_features(clean: np.ndarray,noisy: np.ndarray,noise_std: float,) -> dict:

    filt = ImageFilter()

    filters = {
        "gaussian":  filt.gaussian(noisy),
        "median":    filt.median(noisy),
        "bilateral": filt.bilateral(noisy)}

    filter_scores = {}
    for fname, filtered in filters.items():
        psnr = peak_signal_noise_ratio(clean, filtered)
        ssim = structural_similarity(clean, filtered)
        filter_scores[f"psnr_{fname}"] = psnr
        filter_scores[f"ssim_{fname}"] = ssim

    best_filter = max(filters.keys(),key=lambda k: filter_scores[f"psnr_{k}"])

    edge_feat = EdgeDetector.extract_features(noisy)

    thresh_feat = Threshold.extract_features(noisy)

    return {
        "noise_std":        noise_std,
                            **filter_scores,
        "canny_density":    edge_feat.canny_density,
        "sobel_mean":       edge_feat.sobel_mean,
        "laplacian_var":    edge_feat.laplacian_var,
        "otsu_level":       thresh_feat.otsu_level,
        "foreground_ratio": thresh_feat.foreground_ratio,
        "best_filter":      best_filter}

def build_dataset(image: np.ndarray,n_samples: int = 300,std_range: tuple[int, int] = (5, 50),) -> pd.DataFrame:

    noise_gen = NoiseGenerator()
    records = []

    for i in range(n_samples):
        std = float(np.random.randint(*std_range))
        noisy = noise_gen.gaussian(image, std=std)
        row = extract_sample_features(image, noisy, noise_std=std)
        records.append(row)

        if (i + 1) % 50 == 0:
            log.info(f"  Built {i+1}/{n_samples} samples...")

    return pd.DataFrame(records)

# 9. ML experiment — three models compared
FEATURE_COLS = [
    "noise_std",
    "psnr_gaussian", "ssim_gaussian",
    "psnr_median",   "ssim_median",
    "psnr_bilateral","ssim_bilateral",
    "canny_density", "sobel_mean", "laplacian_var",
    "otsu_level",    "foreground_ratio"]

def run_ml_experiment(df: pd.DataFrame) -> None:

    X = df[FEATURE_COLS]
    y = df["best_filter"]

    log.info(f"\nClass distribution:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)":           SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n" + "="*55)
    print("MODEL COMPARISON")
    print("="*55)

    best_name, best_model, best_cv = None, None, -1

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        results[name] = {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "test_acc": test_acc}

        print(f"\n{name}")
        print(f" CV accuracy : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f" Test accuracy: {test_acc:.3f}")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))

        if cv_scores.mean() > best_cv:
            best_cv = cv_scores.mean()
            best_name = name
            best_model = model

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        importances = importances.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#1D9E75" if v >= importances.median() else "#9FE1CB" for v in importances]
        ax.barh(importances.index, importances.values, color=colors)
        ax.set_xlabel("Feature importance")
        ax.set_title(f"Feature importances — {best_name} (best CV model)")
        ax.axvline(importances.median(), color="gray", linestyle="--", linewidth=0.8, label="median")
        ax.legend()
        plt.tight_layout()
        plt.show()

    summary = pd.DataFrame(results).T
    print("\nSummary:\n", summary.to_string())

# 10. Visualisation
def plot_grid(images: list[np.ndarray],titles: list[str],cmap: str = "gray",) -> None:
    n = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# 11. Section runners
def run_section(section: str, gray: np.ndarray, color: np.ndarray) -> None:
    noise = NoiseGenerator()
    filt  = ImageFilter()
    edge  = EdgeDetector()
    seg   = Segmentation()

    if section == "1":
        plot_grid(
            [gray,
             noise.gaussian(gray),
             noise.salt_and_pepper(gray),
             noise.impulse(gray)],
            ["Original", "Gaussian noise", "Salt & pepper", "Impulse noise"])

    elif section == "2":
        plot_grid(
            [gray,
             filt.gaussian(gray), filt.average(gray), filt.box(gray),
             filt.median(gray),   filt.bilateral(gray),
             filt.non_local_means(gray), filt.wiener(gray)],
            ["Original", "Gaussian", "Average", "Box",
             "Median", "Bilateral", "Non-local means", "Wiener"])

    elif section == "3":
        ef = edge.extract_features(gray)
        log.info(f"Edge features: {ef}")
        plot_grid(
            [gray,
             edge.sobel(gray), edge.prewitt(gray),
             edge.canny(gray), edge.laplacian(gray)],
            ["Original", "Sobel", "Prewitt", "Canny", "Laplacian"])

    elif section == "4":
        _, otsu_val = Threshold.otsu(gray)
        tf = Threshold.extract_features(gray)
        log.info(f"Threshold features: {tf}")
        plot_grid(
            [gray,
             Threshold.binary(gray),
             Threshold.adaptive(gray),
             Threshold.otsu(gray)[0]],
            ["Original", "Binary", "Adaptive", f"Otsu (t={otsu_val:.2f})"])

    elif section == "5":
        plot_grid(
            [gray,
             Segmentation.morphology(gray),
             Segmentation.kmeans(color)],
            ["Original (gray)", "Morphology", "K-means (colour)"])

# 12. entry point
def main() -> None:

    IMAGE_PATH = "sample.jpeg"

    # Choose!
    SECTION = "1"      # 1=noise, 2=filters, 3=edges, 4=threshold, 5=segmentation
    RUN_EVAL = True
    RUN_ML = True
    N_SAMPLES = 300

    path = Path(IMAGE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"cv2.imread failed on {path}.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Section
    if SECTION:
        run_section(SECTION, gray_image, image_rgb)

    # Evaluation
    if RUN_EVAL:
        noise = NoiseGenerator()

        for noisy, label in [
            (noise.gaussian(gray_image), "Gaussian noise"),
            (noise.salt_and_pepper(gray_image), "Salt & pepper"),
            (noise.impulse(gray_image), "Impulse noise")]:
            print(evaluate(gray_image, noisy, name=label))

    # Machine Learning
    if RUN_ML:
        log.info(f"Building dataset ({N_SAMPLES} samples)...")
        df = build_dataset(gray_image, n_samples=N_SAMPLES)
        run_ml_experiment(df)

if __name__ == "__main__":
    main()