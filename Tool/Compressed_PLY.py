import os
import zipfile

base_dir = "dtu"
scan_list = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
output_zip = "ARF+CBAM_epoch15.zip"

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for scan in scan_list:
        scan_folder = os.path.join(base_dir, f"scan{scan}")
        fused_file = os.path.join(scan_folder, "fused.ply")
        if os.path.exists(fused_file):
            print(fused_file)
            zipf.write(fused_file, os.path.relpath(fused_file, base_dir))
        else:
            print(f"Warning: {fused_file} not found")

print(f"Compression completed: {output_zip}")