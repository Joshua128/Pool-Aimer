import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from pool_isolate import mask_better
from modelTraining import circleCoords

canny_edges = None 

@st.cache_resource
def load_model():
    return YOLO("poolWeights.pt")


def solve_quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return root1, root2
    elif discriminant == 0:
        root1 = (-b) / (2 * a)
        return root1, float("inf")
    else:
        return float("inf"), float("inf")


def generate_line(slope, centerx, centery, rad, xroot, pool_center_x, pool_center_y, intercept):
    new_centery = centery + slope * xroot
    new_centerx = centerx + xroot

    line_slope = (pool_center_y - new_centery) / (pool_center_x - new_centerx + 1e-6)
    line_intercept = new_centery - (line_slope * new_centerx)
    return (line_slope, line_intercept, new_centerx, new_centery)


def solve_equations(center_of_white, img, m, b, center_arrays):
    roots = []
    new_radius_points = []

    for h, k, r in center_arrays:
        Xs = (center_of_white[0] - h)
        Ys = (center_of_white[1] - k)

        A = 1 + m**2
        B = (2 * m * Ys) + (2 * Xs)
        C = Xs**2 + Ys**2 - 4 * (r**2)

        root1, root2 = solve_quadratic(A, B, C)

        root1_full = center_of_white[0] + root1
        root2_full = center_of_white[0] + root2

        y_end1 = (root1_full * m + b)
        y_end2 = (root2_full * m + b)

        # NOTE: keeping your original hard-coded bounds (700x800)
        ok1 = (y_end1 < 800 and y_end1 >= 0 and 0 <= root1_full <= 700)
        ok2 = (y_end2 < 800 and y_end2 >= 0 and 0 <= root2_full <= 700)

        if ok1 or ok2:
            root1_full_int = int(root1_full) if np.isfinite(root1_full) else -10**9
            root2_full_int = int(root2_full) if np.isfinite(root2_full) else -10**9

            correct_root = root1 if abs(center_of_white[0] - root1_full_int) < abs(center_of_white[0] - root2_full_int) else root2
            if not np.isfinite(correct_root):
                roots.append((root1, root2))
                continue

            correct_root = int(correct_root)

            line_slope, line_intercept, x2, y2 = generate_line(
                m,
                centerx=center_of_white[0],
                centery=center_of_white[1],
                rad=r,
                xroot=correct_root,
                pool_center_x=h,
                pool_center_y=k,
                intercept=b,
            )

            x2 = int(x2)
            y2 = int(y2)
            new_radius_points.append([line_slope, line_intercept, x2, y2])

            cv2.circle(img, (x2, y2), 8, (0, 255, 0), 3)
            roots.append((root1, root2))

    return {"roots": roots, "radius_points": new_radius_points}


def drawLine(img, lines):
    for line_slope, line_intercept, x, y in lines:
        if line_slope != float("inf") and abs(line_slope) < 100:
            x_beg = ((800 - line_intercept) / (line_slope))
            if x_beg == float("inf"):
                cv2.line(img, (x, 800), (x, 0), (0, 255, 255), 3)
                break
            x_beg = int(x_beg)
            begin_point = (x_beg, 800)
            y_endp = int((line_intercept * -1) / line_slope)
            y_ending = (y_endp, 0)
            cv2.line(img, begin_point, y_ending, (255, 0, 0), 3)
        else:
            cv2.line(img, (x, 800), (x, 0), (0, 255, 255), 3)


def run_logic(img_bgr, model, frame_count, cached_circles):
    """
    Runs your pasted logic on ONE image.
    Returns: annotated_img_bgr, new_frame_count, new_cached_circles
    """
    img = img_bgr.copy()

    mask = mask_better(img)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if frame_count % 30 == 0 or cached_circles is None:
        cached_circles = circleCoords(img, model)

    circles = cached_circles
    center_array = []

    if circles is not None:
        for x, y, w, h in circles:
            rad = int((w - x) / 2)
            center_array.append((x + rad, y + rad, rad))
            cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
            cv2.circle(mask_gray, (x + rad, y + rad), rad, (0, 0, 255), 2)

    _, bright_mask = cv2.threshold(mask_gray, 220, 255, cv2.THRESH_BINARY)

    min_intensity = float("-inf")
    center_of_white = (0, 0, 0)

    for center in center_array:
        x1, x2 = max(0, center[0] - 100), min(700, center[0] + 100)
        y1, y2 = max(0, center[1] - 100), min(800, center[1] + 100)
        if x2 <= x1 or y2 <= y1:
            continue

        x = int(center[0])
        y = int(center[1])
        rad = int(center[2])

        average_intensity = float(np.mean(bright_mask[y1:y2, x1:x2]))
        if average_intensity and average_intensity > min_intensity:
            min_intensity = average_intensity
            center_of_white = (x, y, rad)

    filtered_centers = [
        c for c in center_array
        if (c[0] >= center_of_white[0] + 3 or c[0] <= center_of_white[0] - 3)
    ]

    cv2.circle(img, (center_of_white[0], center_of_white[1]), 13, (0, 0, 255), 3)

    _, thresh_mask = cv2.threshold(bright_mask, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    thresh_mask = thresh_mask[center_of_white[1] - 200: center_of_white[1] + 200, center_of_white[0] - 200: center_of_white[0] + 200]
    thresh_mask = cv2.GaussianBlur(thresh_mask, (5,5), 0)
    thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, np.ones((5,5)))
    global canny_edges
    canny_edges = cv2.Canny(thresh_mask, 25, 80)
    lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=40, minLineLength=10, maxLineGap=20)

    if lines is not None:
        def line_length(line):
            x1, y1, x2, y2 = line[0]
            return np.hypot(x2 - x1, y2 - y1)

        sorted_lines = sorted(lines, key=line_length, reverse=True)
        longest_line_1 = sorted_lines[0][0]
        longest_line_2 = sorted_lines[1][0] if len(sorted_lines) > 1 else None

        x1a, y1a, x2a, y2a = longest_line_1
        x1a += center_of_white[0] - 200 
        y1a += center_of_white[1] - 200
        x2a += center_of_white[0] - 200
        y2a += center_of_white[1] - 200
        if longest_line_2 is not None:
            x1b, y1b, x2b, y2b = longest_line_2

            x1b += center_of_white[0] - 200
            y1b += center_of_white[1] - 200
            x2b += center_of_white[0] - 200
            y2b += center_of_white[1] - 200

            slope = (y2a - y1a) / (x2a - x1a + 1e-6)
            intercept = (int(y1a - (x1a * slope)) + int(y1b - (x1b * slope))) // 2

            width = 700
            endpoint_one = (0, intercept)
            y_endpoint = int((width * slope) + intercept)
            endpoint_two = (width, y_endpoint)

            dic_points = solve_equations(center_of_white, img, slope, intercept, filtered_centers)
            x_values = dic_points["roots"]
            radius_points = dic_points["radius_points"]

            drawLine(img, radius_points)

            for root1, root2 in x_values:
                if np.isfinite(root1) and root1 < 700 and root1 > 0:
                    y_end = int(root1 * slope + intercept)
                    if 0 < y_end < 800:
                        cv2.circle(img, (int(root1), y_end), 8, (0, 255, 0), 2)

                if np.isfinite(root2) and root2 < 700 and root2 > 0:
                    y_end = int(root2 * slope + intercept)
                    if 0 < y_end < 800:
                        cv2.circle(img, (int(root2), y_end), 8, (255, 0, 0), 2)

            cv2.line(img, endpoint_two, endpoint_one, (0, 0, 255), 3)

    frame_count = (frame_count + 1) % 30
    return img, frame_count, cached_circles


class VisionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.frame_count = 0
        self.cached_circles = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (700, 800))
        out, self.frame_count, self.cached_circles = run_logic(
            img, self.model, self.frame_count, self.cached_circles
        )
        return out


def main():
    st.set_page_config(page_title="Pool Vision", layout="wide")
    st.title("Pool Vision")

    model = load_model()

    tab1, tab2 = st.tabs(["Live camera (WebRTC)", "Take a picture (better for cue stick)"])

    with tab1:
        st.caption("Live stream processing (best on desktop; may miss the cue if motion blur / exposure changes).")
        webrtc_streamer(
            key="pool-cam",
            video_transformer_factory=VisionTransformer,
            media_stream_constraints={
                "video": {"facingMode": "environment"},  # back camera on phones
                "audio": False,
            },
            video_html_attrs={
                "autoPlay": True,
                "playsInline": True,   # important on iOS
                "muted": True,
            },
            async_processing=True,
        )

    with tab2:
        st.caption("Take a still photo OR upload one, then we run the SAME logic once.")

        c1, c2 = st.columns(2)

        with c1:
            photo = st.camera_input("Capture a photo")

        with c2:
            upload = st.file_uploader(
                "Upload a photo",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False
            )

        # pick whichever the user provided (camera takes priority if both exist)
        image_source = photo if photo is not None else upload

        if image_source is not None:
            file_bytes = np.asarray(bytearray(image_source.getvalue()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            

            if img_bgr is None:
                st.error("Could not decode the image. Try a different file.")
            else:
                img_bgr = cv2.resize(img_bgr, (700, 800))
                # Force fresh circle detection for still images
                out_bgr, _, _ = run_logic(img_bgr, model, frame_count=0, cached_circles=None)

                st.subheader("Processed result")
                st.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                with st.expander("Show original"):
                    #st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    #st.image(cv2.cvtColor(canny_edges, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    if canny_edges is None:
                        st.info("No canny edges computed yet.")
                    else:
                        st.image(canny_edges, clamp=True, use_container_width=True)


if __name__ == "__main__":
    main()

