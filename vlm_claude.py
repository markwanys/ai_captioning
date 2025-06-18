import anthropic
import base64
import os
import dotenv

dotenv.load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

with open("data/test_cocoa_image.png", "rb") as f:
    img_bytes = f.read()

image_base64 = base64.b64encode(img_bytes).decode("utf-8")

response = client.messages.create(
    model="claude-3-opus-20240229",  # or claude-3-sonnet
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                {"type": "text", "text": """
                You are a crop scientist. Analyze this cocoa farm image and provide structured JSON with:
                - weeding_status
                - tree_condition
                - soil_visibility
                - pest_or_disease_signs
                """}
            ]
        }
    ]
)

print(response.content[0].text)
