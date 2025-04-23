import os
import cv2
import numpy as np
from albumentations import (
    Compose, Rotate, IAAPiecewiseAffine
)
import random
from tqdm import trange

##########################
# To change:
# Pcomm come from ICA and not MCA way before junction with ACA and MCA
# ICA branches into ACA and MCA
# For TopCoW don't have annotations for: 
#       - VA
#       - any from "New vessels"
# 
##########################



# Output directory
PATH = 'datasets/circle_of_willis_TopCoW/'
os.makedirs(PATH, exist_ok=True)

def draw_vessel(img, pt1, pt2, thickness, color):
    """
    Draws a vessel segment (line) between two points.
    """
    cv2.line(img, pt1, pt2, color, thickness)

for i in trange(100):
    # Create a blank image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Center reference
    cx, cy = 256, 256

    # Color dictionary (OpenCV uses BGR)
    COLORS = {
        "ICA":   (0, 0, 255),      # Internal Carotid - Red
        "ACA":   (0, 255, 0),      # Anterior Cerebral - Green
        "PCA":   (255, 0, 0),      # Posterior Cerebral - Blue
        "AComm": (0, 255, 255),    # Anterior Communicating - Yellow-ish
        "PComm": (0, 165, 255),    # Posterior Communicating - Orange
        "MCA":   (255, 255, 0),    # Middle Cerebral - Cyan-ish
        "BA":    (128, 0, 128),    # Basilar - Purple
        # "VA":    (147, 20, 255),   # Vertebral - Pink
        # # New vessels:
        # "SCA":   (255, 255, 255),    # Superior Cerebellar - White
        # "AICA":  (30, 105, 210),   # Anterior Inferior Cerebellar - Chocolate
        # "PICA":  (139, 0, 139),    # Posterior Inferior Cerebellar - Dark Magenta
        # "ASA":   (128, 128, 0),    # Anterior Spinal - Teal
        # "AChA":  (255, 0, 255),    # Anterior Choroidal - Magenta
        # "RAH":   (203, 192, 255)   # Recurrent Artery of Heubner - Pinkish
    }

    # -----------------------------------------------------------------
    # 1) MAIN CIRCLE OF WILLIS: ICA, MCA, ACA, AComm, Basilar, PCA, etc.
    # -----------------------------------------------------------------

    # -------------------- ICA (vertical trunk) + branches --------------------
    # Left ICA
    ICA_left_bottom = (
        cx - 60 + random.randint(-5, 5),
        cy + 120 + random.randint(-5, 5)
    )
    ICA_left_mid = (
        cx - 60 + random.randint(-5, 5),
        cy + 60 + random.randint(-5, 5)
    )
    # Right ICA
    ICA_right_bottom = (
        cx + 60 + random.randint(-5, 5),
        cy + 120 + random.randint(-5, 5)
    )
    ICA_right_mid = (
        cx + 60 + random.randint(-5, 5),
        cy + 60 + random.randint(-5, 5)
    )

    # MCA: Lateral from ICA mid
    MCA_left_end = (
        cx - 120 + random.randint(-5, 5),
        cy + 60  + random.randint(-5, 5)
    )
    MCA_right_end = (
        cx + 120 + random.randint(-5, 5),
        cy + 60  + random.randint(-5, 5)
    )

    # ACA: Superior from ICA mid
    ACA_left_end = (
        cx - 60 + random.randint(-5, 5),
        cy - 20 + random.randint(-5, 5)
    )
    ACA_right_end = (
        cx + 60 + random.randint(-5, 5),
        cy - 20 + random.randint(-5, 5)
    )

    # AComm: Connect left/right ACA
    AComm_mid = (
        cx + random.randint(-3, 3),
        (cy - 20) + random.randint(-3, 3)
    )

    # Basilar artery (vertical trunk)
    BA_top = (
        cx + random.randint(-3, 3),
        (cy + 120) + random.randint(-3, 3)
    )
    BA_bottom = (
        cx + random.randint(-3, 3),
        (cy + 180) + random.randint(-3, 3)
    )

    # Posterior Cerebral Arteries (PCA): branch from top of basilar
    PCA_left_end = (
        cx - 60 + random.randint(-5, 5),
        cy + 80 + random.randint(-5, 5)
    )
    PCA_right_end = (
        cx + 60 + random.randint(-5, 5),
        cy + 80 + random.randint(-5, 5)
    )

    # Posterior Communicating (PComm): connect ICA mid to mid-PCA
    # We'll define a mid-PCA point for each side:
    PCA_left_mid = (
        int((BA_top[0] + PCA_left_end[0]) / 2) + random.randint(-5, 5),
        int((BA_top[1] + PCA_left_end[1]) / 2) + random.randint(-5, 5)
    )
    PCA_right_mid = (
        int((BA_top[0] + PCA_right_end[0]) / 2) + random.randint(-5, 5),
        int((BA_top[1] + PCA_right_end[1]) / 2) + random.randint(-5, 5)
    )

    # Vertebral Arteries from bottom of basilar
    VA_left_end = (
        cx - 20 + random.randint(-5, 5),
        cy + 220 + random.randint(-5, 5)
    )
    VA_right_end = (
        cx + 20 + random.randint(-5, 5),
        cy + 220 + random.randint(-5, 5)
    )

    # -----------------------------------------------------------------
    # 2) ADDITIONAL ARTERIES
    #    SCA, AICA, PICA, ASA, AChA, Recurrent Artery of Heubner
    # -----------------------------------------------------------------

    # --- 2.1 Superior Cerebellar Arteries (SCA) ---
    # Typically branch near the top of the basilar, just below the PCA.
    SCA_left_origin = (
        BA_top[0] - 10 + random.randint(-3, 3),
        BA_top[1] + random.randint(-3, 3)
    )
    SCA_left_end = (
        SCA_left_origin[0] - 30 + random.randint(-5, 5),
        SCA_left_origin[1] - 20 + random.randint(-5, 5)
    )
    SCA_right_origin = (
        BA_top[0] + 10 + random.randint(-3, 3),
        BA_top[1] + random.randint(-3, 3)
    )
    SCA_right_end = (
        SCA_right_origin[0] + 30 + random.randint(-5, 5),
        SCA_right_origin[1] - 20 + random.randint(-5, 5)
    )

    # --- 2.2 Anterior Inferior Cerebellar Arteries (AICA) ---
    # Often branch from the basilar artery around its mid portion.
    # Let's define a mid-Basilar point:
    BA_mid = (
        int((BA_top[0] + BA_bottom[0]) / 2) + random.randint(-2, 2),
        int((BA_top[1] + BA_bottom[1]) / 2) + random.randint(-2, 2)
    )
    AICA_left_origin = (
        BA_mid[0] + random.randint(-5, -3),
        BA_mid[1] + random.randint(-2, 2)
    )
    AICA_left_end = (
        AICA_left_origin[0] - 30 + random.randint(-5, 5),
        AICA_left_origin[1] + 20 + random.randint(-5, 5)
    )
    AICA_right_origin = (
        BA_mid[0] + random.randint(3, 5),
        BA_mid[1] + random.randint(-2, 2)
    )
    AICA_right_end = (
        AICA_right_origin[0] + 30 + random.randint(-5, 5),
        AICA_right_origin[1] + 20 + random.randint(-5, 5)
    )

    # --- 2.3 Posterior Inferior Cerebellar Arteries (PICA) ---
    # Typically arise from the vertebral arteries.
    PICA_left_origin = (
        VA_left_end[0] + random.randint(-2, 2),
        VA_left_end[1] - random.randint(15, 25)  # slightly above VA end
    )
    PICA_left_end = (
        PICA_left_origin[0] - 20 + random.randint(-5, 5),
        PICA_left_origin[1] + 10 + random.randint(-5, 5)
    )
    PICA_right_origin = (
        VA_right_end[0] + random.randint(-2, 2),
        VA_right_end[1] - random.randint(15, 25)
    )
    PICA_right_end = (
        PICA_right_origin[0] + 20 + random.randint(-5, 5),
        PICA_right_origin[1] + 10 + random.randint(-5, 5)
    )

    # --- 2.4 Anterior Spinal Artery (ASA) ---
    # Arises near vertebral arteries (midline). We'll define a short vertical line.
    ASA_top = (
        int((VA_left_end[0] + VA_right_end[0]) / 2) + random.randint(-2, 2),
        int((VA_left_end[1] + VA_right_end[1]) / 2) + random.randint(-2, 2)
    )
    ASA_bottom = (
        ASA_top[0] + random.randint(-3, 3),
        ASA_top[1] + 50 + random.randint(-5, 5)  # extends downward
    )

    # --- 2.5 Anterior Choroidal Arteries (AChA) ---
    # Typically branch from the ICA, near or just distal to the PComm origin.
    # We'll attach them near ICA_mid or slightly above on each side.
    AChA_left_start = (
        ICA_left_mid[0] + random.randint(-3, 3),
        ICA_left_mid[1] + random.randint(-5, -2)
    )
    AChA_left_end = (
        AChA_left_start[0] - 20 + random.randint(-5, 5),
        AChA_left_start[1] - 20 + random.randint(-5, 5)
    )
    AChA_right_start = (
        ICA_right_mid[0] + random.randint(-3, 3),
        ICA_right_mid[1] + random.randint(-5, -2)
    )
    AChA_right_end = (
        AChA_right_start[0] + 20 + random.randint(-5, 5),
        AChA_right_start[1] - 20 + random.randint(-5, 5)
    )

    # --- 2.6 Recurrent Artery of Heubner (RAH) ---
    # Usually branches from ACA (A1 segment) near the AComm region.
    # We'll define a short line from near the ACA -> some lateral point.
    RAH_left_start = (
        int((ACA_left_end[0] + AComm_mid[0]) / 2) + random.randint(-3, 3),
        int((ACA_left_end[1] + AComm_mid[1]) / 2) + random.randint(-3, 3)
    )
    RAH_left_end = (
        RAH_left_start[0] - 15 + random.randint(-3, 3),
        RAH_left_start[1] + 10 + random.randint(-3, 3)
    )
    RAH_right_start = (
        int((ACA_right_end[0] + AComm_mid[0]) / 2) + random.randint(-3, 3),
        int((ACA_right_end[1] + AComm_mid[1]) / 2) + random.randint(-3, 3)
    )
    RAH_right_end = (
        RAH_right_start[0] + 15 + random.randint(-3, 3),
        RAH_right_start[1] + 10 + random.randint(-3, 3)
    )

    # -----------------------------------------------------------------
    # DRAW MAIN CIRCLE
    # -----------------------------------------------------------------
    # Left ICA
    draw_vessel(img, ICA_left_bottom, ICA_left_mid, 8, COLORS["ICA"])
    draw_vessel(img, ICA_left_mid, MCA_left_end, 8, COLORS["MCA"])
    draw_vessel(img, ICA_left_mid, ACA_left_end, 5, COLORS["ACA"])
    
    # Right ICA
    draw_vessel(img, ICA_right_bottom, ICA_right_mid, 8, COLORS["ICA"])
    draw_vessel(img, ICA_right_mid, MCA_right_end, 8, COLORS["MCA"])
    draw_vessel(img, ICA_right_mid, ACA_right_end, 5, COLORS["ACA"])

    # AComm
    draw_vessel(img, ACA_left_end, AComm_mid, 3, COLORS["AComm"])
    draw_vessel(img, ACA_right_end, AComm_mid, 3, COLORS["AComm"])

    # Basilar
    draw_vessel(img, BA_top, BA_bottom, 10, COLORS["BA"])

    # PCA: from BA_top to each side
    draw_vessel(img, BA_top, PCA_left_end, 5, COLORS["PCA"])
    draw_vessel(img, BA_top, PCA_right_end, 5, COLORS["PCA"])
    # PComm: from ICA mid to PCA mid
    draw_vessel(img, ICA_left_mid, PCA_left_mid, 4, COLORS["PComm"])
    draw_vessel(img, PCA_left_mid, PCA_left_end, 4, COLORS["PCA"])  # Connect mid->end
    draw_vessel(img, ICA_right_mid, PCA_right_mid, 4, COLORS["PComm"])
    draw_vessel(img, PCA_right_mid, PCA_right_end, 4, COLORS["PCA"])

    # # Vertebral
    # draw_vessel(img, BA_bottom, VA_left_end, 6, COLORS["VA"])
    # draw_vessel(img, BA_bottom, VA_right_end, 6, COLORS["VA"])

    # # -----------------------------------------------------------------
    # # DRAW NEW VESSELS
    # # -----------------------------------------------------------------
    # # 1) Superior Cerebellar Arteries (SCA)
    # draw_vessel(img, SCA_left_origin, SCA_left_end, 4, COLORS["SCA"])
    # draw_vessel(img, SCA_right_origin, SCA_right_end, 4, COLORS["SCA"])

    # # 2) Anterior Inferior Cerebellar Arteries (AICA)
    # draw_vessel(img, AICA_left_origin, AICA_left_end, 4, COLORS["AICA"])
    # draw_vessel(img, AICA_right_origin, AICA_right_end, 4, COLORS["AICA"])

    # # 3) Posterior Inferior Cerebellar Arteries (PICA)
    # draw_vessel(img, PICA_left_origin, PICA_left_end, 4, COLORS["PICA"])
    # draw_vessel(img, PICA_right_origin, PICA_right_end, 4, COLORS["PICA"])

    # # 4) Anterior Spinal Artery (ASA)
    # draw_vessel(img, ASA_top, ASA_bottom, 3, COLORS["ASA"])

    # # 5) Anterior Choroidal Arteries (AChA)
    # draw_vessel(img, AChA_left_start, AChA_left_end, 3, COLORS["AChA"])
    # draw_vessel(img, AChA_right_start, AChA_right_end, 3, COLORS["AChA"])

    # # 6) Recurrent Artery of Heubner (RAH)
    # draw_vessel(img, RAH_left_start, RAH_left_end, 2, COLORS["RAH"])
    # draw_vessel(img, RAH_right_start, RAH_right_end, 2, COLORS["RAH"])

    # -----------------------------------------------------------------
    # AUGMENTATIONS
    # -----------------------------------------------------------------
    aug = Compose([
        IAAPiecewiseAffine(scale=(0.02, 0.05),
                           nb_rows=4, nb_cols=4,
                           order=1, cval=0, mode='constant', p=1),
        Rotate(limit=30, p=0.5)
    ], p=1)
    img = aug(image=img)['image']

    # Save the image
    out_path = os.path.join(PATH, f"{i:05d}.png")
    cv2.imwrite(out_path, img)
