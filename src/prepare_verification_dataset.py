import os
import shutil
from collections import defaultdict

SOURCE_DIR = "data/signatures/processed"
OUTPUT_DIR = "data/verification"

GENUINE_DIR = os.path.join(OUTPUT_DIR, "genuine")
FORGERY_DIR = os.path.join(OUTPUT_DIR, "forgery")

os.makedirs(GENUINE_DIR, exist_ok=True)
os.makedirs(FORGERY_DIR, exist_ok=True)

# Step 1: Group images by user ID (from filename)
user_groups = defaultdict(list)

for file in os.listdir(SOURCE_DIR):
    if file.endswith(".png"):
        # filename example: original_10_1.png → user = 10
        parts = file.split("_")
        user_id = parts[1]
        user_groups[user_id].append(file)

print(f"Found {len(user_groups)} users")

# Step 2: Create Genuine & Forgery dataset
for user_id, files in user_groups.items():
    for f in files:
        src_path = os.path.join(SOURCE_DIR, f)

        # Copy as genuine
        dst_genuine = os.path.join(GENUINE_DIR, f"user{user_id}_{f}")
        shutil.copy(src_path, dst_genuine)

        # Create forgery using OTHER users' signatures
        for other_user, other_files in user_groups.items():
            if other_user != user_id:
                # take only 1 forgery per other user (keeps dataset balanced)
                forgery_file = other_files[0]
                src_forgery = os.path.join(SOURCE_DIR, forgery_file)

                dst_forgery = os.path.join(
                    FORGERY_DIR,
                    f"forgery_of_user{user_id}_{forgery_file}"
                )
                shutil.copy(src_forgery, dst_forgery)

print("Verification dataset prepared successfully!")