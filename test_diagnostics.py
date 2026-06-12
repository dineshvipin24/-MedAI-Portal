import os
from PIL import Image
from server import analyze_image_pathology

def test_image(img_path):
    print(f"\nEvaluating: {img_path}")
    if not os.path.exists(img_path):
        print("❌ File does not exist!")
        return
    img = Image.open(img_path)
    filename = os.path.basename(img_path)
    
    # Run modality classifier and pathology checks
    # The modality is auto-corrected inside vision_scan, here we pass brain as baseline
    res = analyze_image_pathology(img, modality='brain', filename=filename)
    print(f"🏁 FINAL DIAGNOSIS: {res}")

desktop = r"C:\Users\Vipin\OneDrive\Desktop"
test_image(os.path.join(desktop, "brain.jpeg"))
test_image(os.path.join(desktop, "brain2.jpeg"))
