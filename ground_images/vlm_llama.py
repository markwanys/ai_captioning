from transformers import pipeline

# Create the pipeline for vision + text to text generation
pipe = pipeline(
    "image-text-to-text",
    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    device=0  # Use -1 if you want to run on CPU
)

# Define your prompt with <image> placeholder for image input
prompt = """
<image>
You are a crop scientist. Analyze this cocoa farm image and respond ONLY in JSON format with the following keys:
{
  "weeding_status": "<none|light|moderate|heavy>",
  "tree_condition": "<healthy|yellowing|sparse|diseased>",
  "soil_visibility": "<bare|mulched|overgrown>",
  "pest_or_disease_signs": "<none|present|unsure>"
}
"""

# Load image bytes
with open("data/test_cocoa_image.png", "rb") as f:
    image_bytes = f.read()

# Run the pipeline
result = pipe({"image": image_bytes, "text": prompt})

# Output result
print(result[0]['generated_text'])
