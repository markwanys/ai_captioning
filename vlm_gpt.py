# 2. Import necessary libraries
import openai
from PIL import Image
import base64
import io
import dotenv

# 3. Set your OpenAI API key securely
import os

dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Read image and encode as base64
image_path = "data/test_cocoa_image.png"
with open(image_path, "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Vision prompt
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """
                You are a crop scientist. Analyze the cocoa farm image and respond in structured JSON.
                Include:
                - weeding status
                - tree condition
                - soil visibility
                - pest or disease signs
                """ },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    temperature=0.3,
)

# Print result
print(response.choices[0].message.content)
