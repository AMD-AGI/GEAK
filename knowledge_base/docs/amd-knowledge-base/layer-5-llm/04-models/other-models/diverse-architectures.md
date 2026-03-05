# Other Model Architectures on ROCm

*Comprehensive guide to running diverse AI model architectures on AMD GPUs*

## Overview

Beyond the popular transformer models like GPT, LLaMA, and Mistral, there are many other important AI architectures that can be optimized for AMD GPUs. This guide covers deployment and optimization of various model types including vision transformers, diffusion models, retrieval systems, and emerging architectures.

## Vision Models

### Vision Transformers (ViT)
```python
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

class ViTROCm:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to("cuda")
        
    def classify_image(self, image_path_or_url):
        # Load image
        if image_path_or_url.startswith('http'):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get top predictions
        top5_indices = predictions.argsort(descending=True)[0][:5]
        
        results = []
        for idx in top5_indices:
            label = self.model.config.id2label[idx.item()]
            score = predictions[0][idx].item()
            results.append({"label": label, "score": score})
            
        return results

# Usage
vit_classifier = ViTROCm()
results = vit_classifier.classify_image("path/to/image.jpg")
```

### CLIP (Contrastive Language-Image Pre-Training)
```python
import torch
import clip
from PIL import Image

class CLIPROCm:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
    def encode_text(self, texts):
        """Encode text descriptions"""
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image_path):
        """Encode image"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def zero_shot_classification(self, image_path, class_names):
        """Perform zero-shot image classification"""
        # Encode image
        image_features = self.encode_image(image_path)
        
        # Encode class names as text prompts
        text_prompts = [f"a photo of a {class_name}" for class_name in class_names]
        text_features = self.encode_text(text_prompts)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).softmax(dim=-1)
        
        # Return results
        results = []
        for i, class_name in enumerate(class_names):
            results.append({
                "class": class_name,
                "probability": similarities[0][i].item()
            })
            
        return sorted(results, key=lambda x: x["probability"], reverse=True)
    
    def image_text_similarity(self, image_path, text_descriptions):
        """Calculate similarity between image and text descriptions"""
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text_descriptions)
        
        similarities = (image_features @ text_features.T).squeeze()
        
        results = []
        for i, text in enumerate(text_descriptions):
            results.append({
                "text": text,
                "similarity": similarities[i].item()
            })
            
        return results

# Usage
clip_model = CLIPROCm()
classes = ["cat", "dog", "car", "airplane", "house"]
results = clip_model.zero_shot_classification("image.jpg", classes)
```

## Diffusion Models

### Stable Diffusion
```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class StableDiffusionROCm:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Use DPM-Solver for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Enable memory efficient attention
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload()
        
        # Move to GPU
        self.pipe = self.pipe.to("cuda")
    
    def generate_image(self, prompt, negative_prompt=None, 
                      num_inference_steps=20, guidance_scale=7.5,
                      height=512, width=512, num_images=1):
        """Generate images from text prompts"""
        
        with torch.autocast("cuda", dtype=torch.float16):
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images,
                generator=torch.Generator(device="cuda").manual_seed(42)
            ).images
        
        return images
    
    def image_to_image(self, init_image, prompt, strength=0.75):
        """Generate variations of an existing image"""
        from diffusers import StableDiffusionImg2ImgPipeline
        
        # Use img2img pipeline
        img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipe.components)
        
        with torch.autocast("cuda", dtype=torch.float16):
            images = img2img_pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=20,
            ).images
        
        return images

# Usage
sd_generator = StableDiffusionROCm()
images = sd_generator.generate_image(
    "A futuristic city with flying cars, digital art style",
    negative_prompt="blurry, low quality",
    num_images=2
)
```

### ControlNet
```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import numpy as np

class ControlNetROCm:
    def __init__(self, controlnet_type="canny"):
        # Load ControlNet model
        if controlnet_type == "canny":
            controlnet_id = "lllyasviel/sd-controlnet-canny"
        elif controlnet_type == "depth":
            controlnet_id = "lllyasviel/sd-controlnet-depth"
        elif controlnet_type == "pose":
            controlnet_id = "lllyasviel/sd-controlnet-openpose"
        
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16
        )
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to("cuda")
        self.controlnet_type = controlnet_type
    
    def preprocess_canny(self, image, low_threshold=100, high_threshold=200):
        """Preprocess image for Canny ControlNet"""
        image_array = np.array(image)
        canny_image = cv2.Canny(image_array, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        return Image.fromarray(canny_image)
    
    def generate_controlled_image(self, control_image, prompt, 
                                num_inference_steps=20):
        """Generate image with ControlNet guidance"""
        
        if self.controlnet_type == "canny":
            control_image = self.preprocess_canny(control_image)
        
        with torch.autocast("cuda", dtype=torch.float16):
            images = self.pipe(
                prompt=prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                controlnet_conditioning_scale=1.0,
                guidance_scale=7.5,
            ).images
        
        return images

# Usage
controlnet = ControlNetROCm("canny")
control_img = Image.open("input_image.jpg")
result = controlnet.generate_controlled_image(
    control_img, 
    "a beautiful landscape painting"
)
```

## Speech and Audio Models

### Whisper (Speech Recognition)
```python
import whisper
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperROCm:
    def __init__(self, model_size="base"):
        # Option 1: Using OpenAI's whisper
        self.whisper_model = whisper.load_model(model_size).to("cuda")
        
        # Option 2: Using HuggingFace transformers
        model_name = f"openai/whisper-{model_size}"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.hf_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to("cuda")
    
    def transcribe_openai(self, audio_path, language=None):
        """Transcribe audio using OpenAI's whisper"""
        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            fp16=True  # Use FP16 on GPU
        )
        return result
    
    def transcribe_hf(self, audio_path):
        """Transcribe audio using HuggingFace transformers"""
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.hf_model.generate(
                inputs.input_features,
                max_new_tokens=448,
                do_sample=False
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def translate_to_english(self, audio_path):
        """Translate non-English audio to English"""
        result = self.whisper_model.transcribe(
            audio_path,
            task="translate",
            fp16=True
        )
        return result["text"]

# Usage
whisper_asr = WhisperROCm("base")
transcription = whisper_asr.transcribe_openai("audio_file.wav")
print(f"Transcription: {transcription['text']}")
```

### Text-to-Speech Models
```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf

class SpeechT5ROCm:
    def __init__(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Load vocoder for audio generation
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Load speaker embeddings
        from datasets import load_dataset
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation"
        )
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to("cuda")
    
    def synthesize_speech(self, text, output_path="output.wav"):
        """Convert text to speech"""
        inputs = self.processor(text=text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embeddings,
                vocoder=self.vocoder
            )
        
        # Save audio
        sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
        return speech.cpu().numpy()

# Usage
tts_model = SpeechT5ROCm()
audio = tts_model.synthesize_speech(
    "Hello, this is a test of text-to-speech on AMD GPU!"
)
```

## Retrieval and Embedding Models

### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np

class EmbeddingSearchROCm:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cuda")
        self.index = None
        self.documents = []
        
    def encode_documents(self, documents, batch_size=32):
        """Encode documents into embeddings"""
        self.documents = documents
        
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda"
        )
        
        # Convert to CPU for FAISS indexing
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        
        # Create FAISS index
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
        
        return embeddings_np
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("No documents indexed. Call encode_documents first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_tensor=True,
            device="cuda"
        ).cpu().numpy().astype('float32')
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "document": self.documents[idx],
                "similarity": float(score),
                "index": int(idx)
            })
        
        return results
    
    def compute_similarity_matrix(self, texts1, texts2):
        """Compute similarity matrix between two sets of texts"""
        embeddings1 = self.model.encode(
            texts1,
            convert_to_tensor=True,
            device="cuda"
        )
        embeddings2 = self.model.encode(
            texts2,
            convert_to_tensor=True,
            device="cuda"
        )
        
        # Compute cosine similarity
        similarity_matrix = torch.nn.functional.cosine_similarity(
            embeddings1.unsqueeze(1),
            embeddings2.unsqueeze(0),
            dim=2
        )
        
        return similarity_matrix.cpu().numpy()

# Usage
embedding_search = EmbeddingSearchROCm()

documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing deals with text and speech",
    "Computer vision focuses on image and video analysis"
]

embedding_search.encode_documents(documents)
results = embedding_search.search("What is AI?", top_k=3)
```

### Dense Passage Retrieval (DPR)
```python
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
import torch

class DPRROCm:
    def __init__(self):
        # Load question encoder
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.q_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Load context encoder
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        self.ctx_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base",
            torch_dtype=torch.float16
        ).to("cuda")
        
    def encode_questions(self, questions):
        """Encode questions into embeddings"""
        inputs = self.q_tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256
        ).to("cuda")
        
        with torch.no_grad():
            embeddings = self.q_encoder(**inputs).pooler_output
        
        return embeddings
    
    def encode_contexts(self, contexts):
        """Encode contexts/passages into embeddings"""
        inputs = self.ctx_tokenizer(
            contexts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256
        ).to("cuda")
        
        with torch.no_grad():
            embeddings = self.ctx_encoder(**inputs).pooler_output
        
        return embeddings
    
    def retrieve_passages(self, question, passages, top_k=5):
        """Retrieve most relevant passages for a question"""
        q_embedding = self.encode_questions([question])
        ctx_embeddings = self.encode_contexts(passages)
        
        # Compute similarities
        similarities = torch.matmul(q_embedding, ctx_embeddings.T)
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities[0], k=min(top_k, len(passages)))
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append({
                "passage": passages[idx.item()],
                "score": score.item(),
                "index": idx.item()
            })
        
        return results

# Usage
dpr = DPRROCm()

passages = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is located in Paris, France.",
    "London is the capital city of England and the United Kingdom.",
    "The Thames is a river that flows through London."
]

question = "What is the capital of France?"
results = dpr.retrieve_passages(question, passages, top_k=2)
```

## Multimodal Models

### BLIP (Bootstrapping Language-Image Pre-training)
```python
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import torch
from PIL import Image

class BLIPROCm:
    def __init__(self):
        # Image captioning model
        self.caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Visual question answering model
        self.vqa_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base",
            torch_dtype=torch.float16
        ).to("cuda")
    
    def generate_caption(self, image_path, max_length=50):
        """Generate caption for an image"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.caption_processor(image, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            out = self.caption_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def answer_question(self, image_path, question):
        """Answer a question about an image"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.vqa_processor(
            image,
            question,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            out = self.vqa_model.generate(
                **inputs,
                max_length=20,
                num_beams=5,
                early_stopping=True
            )
        
        answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)
        return answer
    
    def conditional_captioning(self, image_path, text_prompt):
        """Generate caption conditioned on text prompt"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.caption_processor(
            image,
            text_prompt,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            out = self.caption_model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption

# Usage
blip_model = BLIPROCm()
caption = blip_model.generate_caption("image.jpg")
answer = blip_model.answer_question("image.jpg", "What color is the sky?")
```

## Specialized Architectures

### Graph Neural Networks (GNN)
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
from torch_geometric.data import Data, DataLoader

class GNNROCm:
    def __init__(self, model_type="GCN", num_features=16, num_classes=7):
        self.device = "cuda"
        
        if model_type == "GCN":
            self.model = GCN(num_features, num_classes).to(self.device)
        elif model_type == "GAT":
            self.model = GAT(num_features, num_classes).to(self.device)
        elif model_type == "GraphSAGE":
            self.model = GraphSAGE(num_features, num_classes).to(self.device)
            
        self.model_type = model_type
    
    def train_model(self, data_loader, epochs=200):
        """Train the GNN model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in data_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss / len(data_loader):.4f}')
    
    def predict(self, data):
        """Make predictions on graph data"""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            pred = self.model(data.x, data.edge_index).argmax(dim=1)
        
        return pred.cpu()

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)
        
    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Usage
gnn_model = GNNROCm("GCN", num_features=1433, num_classes=7)
# Training would require graph data in PyTorch Geometric format
```

### Time Series Models
```python
import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformer

class TimeSeriesROCm:
    def __init__(self, sequence_length=100, num_features=1):
        self.device = "cuda"
        self.sequence_length = sequence_length
        
        # LSTM-based model for time series
        self.lstm_model = LSTMTimeSeriesModel(
            input_size=num_features,
            hidden_size=64,
            num_layers=2,
            output_size=1
        ).to(self.device)
        
        # Transformer-based model
        config = TimeSeriesTransformerConfig(
            context_length=sequence_length,
            prediction_length=1,
            num_time_features=num_features
        )
        self.transformer_model = TimeSeriesTransformer(config).to(self.device)
    
    def train_lstm(self, train_loader, epochs=100):
        """Train LSTM model for time series prediction"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        self.lstm_model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.6f}')
    
    def predict_lstm(self, sequence):
        """Predict next values using LSTM"""
        self.lstm_model.eval()
        sequence = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        
        if len(sequence.shape) == 2:
            sequence = sequence.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            prediction = self.lstm_model(sequence)
        
        return prediction.cpu().numpy()

class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        return out

# Usage
ts_model = TimeSeriesROCm(sequence_length=50, num_features=1)
# Training would require proper time series data loader
```

## Model Optimization Techniques

### Dynamic Quantization
```python
import torch.quantization

def optimize_model_quantization(model, example_input):
    """Apply various quantization techniques"""
    
    # Post-training quantization
    model.eval()
    
    # Dynamic quantization (CPU)
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # For GPU, use FP16
    model_fp16 = model.half()
    
    return quantized_model, model_fp16

def benchmark_model_variants(models, example_input, num_runs=100):
    """Benchmark different model variants"""
    results = {}
    
    for name, model in models.items():
        times = []
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(example_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                _ = model(example_input)
            end_time.record()
            
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
        
        results[name] = {
            'avg_time': sum(times) / len(times),
            'std_time': torch.tensor(times).std().item(),
            'throughput': 1000 / (sum(times) / len(times))  # inferences/second
        }
    
    return results
```

### Memory Optimization
```python
import torch.utils.checkpoint as checkpoint

class MemoryOptimizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        # Use gradient checkpointing to save memory
        return checkpoint.checkpoint(self.base_model, x)

def optimize_memory_usage():
    """Various memory optimization techniques"""
    
    # Enable memory efficient attention
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Use mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Memory mapping for large datasets
    class MemoryMappedDataset(torch.utils.data.Dataset):
        def __init__(self, data_path):
            import numpy as np
            self.data = np.memmap(data_path, mode='r', dtype=np.float32)
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return torch.from_numpy(self.data[idx])
    
    return scaler, MemoryMappedDataset
```

## Production Deployment

### Multi-Model Serving
```python
from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI(title="Multi-Model AI API")

# Global model registry
models = {}

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    global models
    
    # Load different model types
    models['vit'] = ViTROCm()
    models['clip'] = CLIPROCm()
    models['whisper'] = WhisperROCm('base')
    models['blip'] = BLIPROCm()

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """Image classification endpoint"""
    image = Image.open(file.file)
    results = models['vit'].classify_image(image)
    return {"predictions": results}

@app.post("/zero-shot-classify")
async def zero_shot_classify(file: UploadFile = File(...), classes: str = ""):
    """Zero-shot image classification"""
    image = Image.open(file.file)
    class_list = classes.split(",")
    results = models['clip'].zero_shot_classification(image, class_list)
    return {"predictions": results}

@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """Audio transcription"""
    # Save uploaded file temporarily
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    result = models['whisper'].transcribe_openai(audio_path)
    
    # Clean up
    import os
    os.remove(audio_path)
    
    return {"transcription": result["text"]}

@app.post("/caption-image")
async def caption_image(file: UploadFile = File(...)):
    """Image captioning"""
    # Save uploaded file temporarily
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    caption = models['blip'].generate_caption(image_path)
    
    # Clean up
    import os
    os.remove(image_path)
    
    return {"caption": caption}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Best Practices

### Model Selection Guidelines
```python
def select_optimal_model(task_type, dataset_size, latency_requirements):
    """Guide for selecting optimal model architecture"""
    
    recommendations = {
        "image_classification": {
            "small_dataset": "ViT-Small with transfer learning",
            "large_dataset": "EfficientNet or ResNet",
            "real_time": "MobileNet or EfficientNet-Lite"
        },
        "text_generation": {
            "short_text": "GPT-2 or small T5",
            "long_text": "GPT-J or LLaMA",
            "code": "CodeGen or CodeT5"
        },
        "multimodal": {
            "image_text": "CLIP or BLIP",
            "video_text": "VideoCLIP",
            "audio_text": "Whisper + text model"
        }
    }
    
    return recommendations.get(task_type, "Contact AI team for specialized architectures")

# Performance monitoring
def monitor_model_performance(model, test_data, metrics=['latency', 'memory', 'accuracy']):
    """Comprehensive model performance monitoring"""
    
    results = {}
    
    if 'latency' in metrics:
        # Measure inference latency
        times = []
        for data in test_data[:100]:  # Sample
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                _ = model(data)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        results['avg_latency_ms'] = sum(times) / len(times)
        results['p95_latency_ms'] = sorted(times)[int(0.95 * len(times))]
    
    if 'memory' in metrics:
        # Memory usage
        torch.cuda.reset_peak_memory_stats()
        
        for data in test_data[:10]:
            with torch.no_grad():
                _ = model(data)
        
        results['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        results['current_memory_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
    
    return results
```

## Resources

### Documentation
- [Hugging Face Model Hub](https://huggingface.co/models)
- [PyTorch Model Zoo](https://pytorch.org/serve/model_zoo.html)
- [Diffusers Library](https://huggingface.co/docs/diffusers)

### Related Guides
- [Memory Optimization](../../03-training/optimization/memory-optimization.md)
- [Production Serving](../../02-inference/deployment/production-serving.md)
- [Custom Kernels](../../05-advanced/custom-kernels/triton-kernels.md)

### Specialized Libraries
- [Sentence Transformers](https://www.sbert.net/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [OpenAI Whisper](https://github.com/openai/whisper)

---
*Tags: computer-vision, nlp, multimodal, diffusion, speech, retrieval, gnn, time-series, rocm*
*Estimated reading time: 75 minutes*