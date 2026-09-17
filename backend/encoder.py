import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Union, List

class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP model and processor.
        Automatically detects GPU if available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.model.config.projection_dim # Should be 512 for ViT-B/32
        
        print("CLIP model loaded successfully.")

    def encode_image(self, image_input: Union[str, Image.Image]) -> List[float]:
        """
        Processes an image (either a file path or PIL Image object) 
        and generates a 512-dim L2-normalized vector.
        """
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # L2 Normalization for Cosine Similarity
            image_features = F.normalize(image_features, p=2, dim=-1)
            return image_features.cpu().numpy().flatten().tolist()
        
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def encode_text(self, text: str) -> List[float]:
        """
        Processes text and generates a 512-dim L2-normalized vector.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # L2 Normalization for Cosine Similarity
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features.cpu().numpy().flatten().tolist()
