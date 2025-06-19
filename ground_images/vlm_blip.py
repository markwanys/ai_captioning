from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import json

# Load the image
image_path = "data/test_cocoa_image.png"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Load the BLIP-2 processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Define your task prompt
# prompt = """
# You are a crop scientist analyzing a ground-level image of a cocoa farm. 
# Describe the following in structured JSON:
# - Weeding status (none, light, moderate, heavy)
# - Tree condition (leaf color, canopy fullness, any signs of disease)
# - Soil condition (bare, mulched, weedy)
# - Optional: pest or disease symptoms
# """

prompt = """
You are an agricultural crop scientist analyzing a cocoa farm image.

Please respond ONLY in the following structured JSON format:

{
  "weeding_status": "<none | light | moderate | heavy>",
  "tree_condition": "<e.g., healthy, yellowing, sparse leaves, etc.>",
  "soil_visibility": "<bare | mulched | covered with weeds>",
  "pest_or_disease_signs": "<none | visible signs | unsure>"
}

Now analyze the image and respond with this JSON structure only, no additional text.
"""



# Prepare inputs
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256)

# Decode and print
response = processor.decode(output[0], skip_special_tokens=True)
print("📋 Model Response:\n", response)

# Try parsing as JSON (if model followed instruction correctly)
try:
    parsed = json.loads(response)
    print("\n✅ Structured Output:")
    for k, v in parsed.items():
        print(f"{k}: {v}")
except:
    print("\n⚠️ Could not parse structured JSON. Raw output shown above.")
