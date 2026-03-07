import cv2
import numpy as np

class DocumentVisionEngine:

    def __init__(self):
        print("[Vision] Engine Ready")

    def remove_grid_lines(self, image):
        """
        Remove notebook/grid lines from image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal lines
        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        # Detect vertical lines
        vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine
        grid_lines = cv2.add(horiz_lines, vert_lines)
        
        # Invert and combine with original
        no_grid = cv2.bitwise_and(gray, cv2.bitwise_not(grid_lines))
        
        return no_grid

    def deskew(self, image):
        """
        Deskew a skewed image by detecting the angle of text lines.
        """
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def perspective_correction(self, img):
        """
        Correct perspective distortion in document images.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return img

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        doc = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                doc = approx
                break

        if doc is None:
            return img

        # Apply perspective transform
        pts = doc.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return warped

    def enhance_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Skip experimental deskew/perspective/grid-removal for now as they are unstable on high-res photos
        # We go straight to clean contrast enhancement
        
        # Step 1: Resize to a standard height for better OCR consistency
        target_height = 800
        h, w = gray.shape[:2]
        if h != target_height:
            scale = target_height / float(h)
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        # Step 2: Denoise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Step 3: High-Quality Contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
        enhanced = clahe.apply(denoised)
        
        # Step 4: Subtle Sharpening
        blur = cv2.GaussianBlur(enhanced, (0,0), 3)
        res = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        
        return res
