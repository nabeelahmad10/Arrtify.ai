from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TextToIMAGEpyfile import generate_image, get_translation

app = FastAPI()  # This is the FastAPI instance that Uvicorn looks for

class TextInput(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


@app.post("/generate/")
async def generate_image_endpoint(input: TextInput):
    try:
        translated_prompt = get_translation(input.text, "en")
        generated_image = generate_image(translated_prompt)
        generated_image.save("generated_image.png")
        return {"message": "Image generated successfully", "file_path": "generated_image.png"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
