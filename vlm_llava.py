import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json

# Load the model and processor
model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load your cocoa farm image
image_path = "data/test_cocoa_image.png"
image = Image.open(image_path).convert("RGB")

# Strict JSON prompt
# prompt = """
# <image>
# You are a crop scientist analyzing a cocoa farm image. Respond with JSON only.

# {
#   "weeding_status": "<none | light | moderate | heavy>",
#   "tree_condition": "<healthy | yellowing | sparse canopy | diseased>",
#   "soil_visibility": "<bare | mulched | overgrown>",
#   "pest_or_disease_signs": "<none | signs of pests | signs of disease | unsure>"
# }
# """

prompt = """
<image>
You are a crop scientist. Analyze the cocoa farm image and respond in structured JSON.
Include:
- weeding status
- tree condition
- soil visibility
- pest or disease signs
"""


# Preprocess
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=512)
output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

print("\n📋 Raw Output:")
print(output)

# Try parsing JSON
try:
    result = json.loads(output)
    print("\n✅ Parsed Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
except Exception as e:
    print("\n⚠️ Could not parse JSON. Error:", e)
