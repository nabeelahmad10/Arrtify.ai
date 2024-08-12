from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from TextToIMAGEpyfile import generate_image, get_translation, calculate_clip_score  # Import your model functions

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/generate/")
async def generate_image_endpoint(input: TextInput):
    try:
        translated_prompt = get_translation(input.text, "en")
        generated_image = generate_image(translated_prompt)

        # Save the generated image to a file
        generated_image.save("generated_image.png")

        return {"message": "Image generated successfully", "file_path": "generated_image.png"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
