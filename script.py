import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(" OPENAI_API_KEY not found in .env file")

IMAGE_FOLDER = "images"           
OUTPUT_PATH = "captions.jsonl"    
MODEL = "gpt-4.1-mini"            
MAX_OUTPUT = 200                  

client = OpenAI(api_key=API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

def caption_image(base64_image):
    prompt = """
Generate two captions for this image:

1. A correct, grounded, accurate caption.
2. A slightly incorrect caption (change ONLY 1–2 details: color, count, object, action).

Return ONLY JSON:
{
  "correct_caption": "...",
  "incorrect_caption": "..."
}
"""

    response = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    return json.loads(response.output_text)

def run():
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_files)} images.\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for img_file in tqdm(image_files, desc="Processing"):
            image_id = os.path.splitext(img_file)[0]
            path = os.path.join(IMAGE_FOLDER, img_file)

            try:
                b64 = encode_image(path)
                captions = caption_image(b64)

                out.write(json.dumps({
                    "image_id": image_id,
                    "correct_caption": captions["correct_caption"],
                    "incorrect_caption": captions["incorrect_caption"]
                }) + "\n")

            except Exception as e:
                print(f"\n Error with {image_id}: {e}")

    print(f"\n Done. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
