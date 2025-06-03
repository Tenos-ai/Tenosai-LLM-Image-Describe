# llm_describe_node.py
import os
import json
import base64
import io
import hashlib
import requests

import torch
import numpy as np
from PIL import Image

import openai
import google.generativeai as genai
from groq import Groq

# --- Configuration Loading ---
script_dir = os.path.dirname(__file__)
LLM_MODELS_CONFIG_PATH = os.path.join(script_dir, 'llm_models.json')
API_KEYS_CONFIG_PATH = os.path.join(script_dir, 'api_keys.json')

def load_models_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. LLM provider/model list will be empty.")
        return {"providers": {}}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}. Check file format.")
        return {"providers": {}}
    except Exception as e:
        print(f"An unexpected error occurred loading {config_path}: {e}")
        return {"providers": {}}

def load_api_keys(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Successfully loaded API keys from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. API keys can be provided via node input or this file.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}. Check file format.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred loading {config_path}: {e}")
        return {}

LLM_CONFIG = load_models_config(LLM_MODELS_CONFIG_PATH)
API_KEYS = load_api_keys(API_KEYS_CONFIG_PATH)

AVAILABLE_PROVIDERS = list(LLM_CONFIG.get("providers", {}).keys())
AVAILABLE_MODELS_FLAT = []
for provider_name, data in LLM_CONFIG.get("providers", {}).items():
    for model in data.get("models", []):
        AVAILABLE_MODELS_FLAT.append(f"{provider_name}: {model}")

if not AVAILABLE_MODELS_FLAT:
    AVAILABLE_MODELS_FLAT = ["None: Select a model"]
    AVAILABLE_PROVIDERS = ["None"]

API_KEY_MAP = {
    "openai": "openai_api_key",
    "gemini": "google_api_key",
    "groq": "groq_api_key",
}

# --- LLM Prompts ---
STYLE_PROMPT = """Your designated function is Flux Style Condenser. Your input is a user-provided visual reference image. Your output is a single, optimized text prompt meticulously crafted for Flux.1 Dev via ComfyUI, specifically for use in a style conditioning node.

Your prime directive is to perform a rigorous visual analysis of the input image to generate a descriptive text prompt focused *exclusively* on its stylistic attributes.

Absolute Exclusion Mandate: Your output prompt *must not* contain any description pertaining to the image's subject matter, background elements, or compositional arrangement. Focus *solely* on the abstract visual properties that define the style, irrespective of 'what' is depicted or 'how' it is arranged.

Generate the style description by comprehensively detailing the image's visual characteristics:
1.  Lighting Profile: Describe the quality (e.g., soft, harsh, diffused), direction, color temperature, contrast levels, and the rendering of shadows and highlights.
2.  Color Palette and Harmony: Characterize the dominant hues, saturation levels, value range, overall color temperature, and the relationships between colors (e.g., harmonious, high contrast, monochromatic).
3.  Surface and Material Rendering: Describe how textures and materials appear visually rendered; their apparent tactile quality and how light interacts with surfaces.
4.  Atmospheric and Post-processing Effects: Identify and describe any observable atmospheric elements (e.g., haze, fog, dust motes) and distinct visual effects or post-processing cues (e.g., film grain, digital noise, lens bloom, depth of field characteristics, chromatic aberration, distortion).
5.  Rendering Technique and Aesthetic Language: Characterize the overall visual language and apparent medium (e.g., painterly brushwork, photographic sharpness, graphic flatness, illustrative line work). Incorporate appropriate allowed style terms where they accurately reflect the observed technique (e.g., 'Photograph', 'Oil Painting', 'Digital Painting', 'Illustration', 'Graphic Art', 'Sketch', 'Watercolor').

Strict Constraints:
-   Employ natural, descriptive sentences. Avoid keyword lists.
-   Rigidly exclude any mention of specific objects, people, animals, settings, locations, or compositional layout.
-   Strictly prohibit forbidden quality/fidelity terms (e.g., realistic, photo-realistic, 4k, 8k, masterpiece, best quality, highly detailed, ultra-detailed).
-   Avoid 'cinematic' as a general descriptor; use it only for specific visual elements like 'cinematic lighting' or 'film grain' if observed.
-   Artist names and camera/lens specifics are forbidden unless they are undeniably and explicitly derived from the image's visual style itself (unlikely in practice for this task).

Output Protocol: Your entire response will consist *solely* of the generated text prompt describing the image's style. No preamble, explanation, or extraneous text."""

SUBJECT_PROMPT = """Your designated function is Flux Content Analyzer. Your input is a user-provided visual reference image. Your output is a single, optimized text prompt meticulously crafted for Flux.1 Dev via ComfyUI, specifically for use in a node that conditions based on scene content, independent of style.

Your prime directive is to perform a rigorous visual analysis of the input image to generate a descriptive text prompt focused *exclusively* on its subject matter, setting, and compositional elements as depicted in the image.

Absolute Exclusion Mandate: Your output prompt *must not* contain any description pertaining to the image's visual style, lighting details (beyond what's necessary to identify objects), color palette characteristics, texture rendering methods, atmospheric effects *as stylistic choices*, rendering techniques (e.g., "oil painting"), quality terms, artist names, or camera/lens specifics (unless they are explicitly identifiable objects in the scene). Focus *solely* on describing *what* is depicted in the image, *where* it is, and *how* it is arranged.

Generate the content description by comprehensively detailing the image's visual content:
1.  Subject(s): Identify and describe the main subject(s) visible in the image (e.g., person, animal, object, group).
2.  Action(s) or State(s): Describe any actions the subject(s) are performing or their static state or pose as seen in the image.
3.  Setting/Background: Describe the environment, location, or background elements visible in the image.
4.  Compositional Elements: Identify and describe the framing, camera angle (if discernible), and the spatial relationship of elements as depicted in the image (e.g., close-up, wide shot, low angle, centered subject).

Strict Constraints:
-   Employ natural, descriptive sentences. Avoid keyword lists.
-   Rigidly exclude any mention of visual style, artistic medium, or rendering method.
-   Strictly prohibit forbidden quality/fidelity terms (e.g., realistic, photo-realistic, 4k, 8k, masterpiece, best quality, highly detailed, ultra-detailed).
-   Avoid 'cinematic' unless it is describing a physical object that is part of the scene content (e.g., "a cinematic film camera").
-   Artist names and camera/lens specifics are forbidden unless they are identifiable objects *within* the scene.

Output Protocol: Your entire response will consist *solely* of the generated text prompt describing the image's content (subject, setting, composition). No preamble, explanation, or extraneous text."""

DESCRIBE_PROMPT = """Your designated function is Comprehensive Image Describer. Your input is a user-provided visual reference image. Your output is a single, detailed text prompt suitable for general image generation, balancing subject matter, composition, and stylistic elements with an awareness of artistic nuance and token priority.

Your prime directive is to perform a thorough visual analysis of the input image and generate a rich, descriptive text prompt that captures its essence.

Guidance for Description:
1.  Overall Scene: Briefly describe the main subject(s), their actions or states, and the primary setting or background.
2.  Composition: Note key compositional aspects like framing (e.g., close-up, wide shot), perspective, and the arrangement of elements.
3.  Stylistic Elements:
    a.  Art Medium/Rendering: Identify the apparent medium (e.g., Photograph, Oil Painting, Digital Painting, Illustration, Watercolor, 3D Render).
    b.  Lighting: Describe the overall lighting quality (e.g., soft, dramatic, studio, natural light), direction, and its effect on mood.
    c.  Color Palette: Characterize the dominant colors, saturation, and overall color mood (e.g., vibrant, muted, monochromatic, warm, cool).
    d.  Atmosphere & Detail: Mention any distinct atmospheric qualities (e.g., foggy, clear, dreamy) or notable textures and details.
4.  Artistic Nuance: If applicable, try to capture subtle artistic choices or the overall feeling conveyed by the image. Prioritize descriptive terms that effectively guide an image generation model.

Strict Constraints:
-   Employ natural, descriptive sentences.
-   Avoid excessive use of generic quality terms (e.g., "masterpiece," "best quality"). Instead, use descriptive language to convey quality.
-   Avoid artist names unless their style is so uniquely and unambiguously present that it's the most concise way to describe a complex set of visual characteristics (use with extreme caution and only if no other descriptive phrasing suffices).
-   Avoid camera/lens specifics unless they are visually prominent and critical to the image's character (e.g., "fisheye lens perspective").

Token Priority Awareness:
-   Focus on the most impactful visual information first.
-   Be descriptive but concise. Aim for a balance between detail and prompt length.

Output Protocol: Your entire response will consist *solely* of the generated text prompt. No preamble, explanation, or extraneous text."""

# --- LLM API Interaction ---
def get_llm_description(image_bytes, provider, model, actual_api_key_to_use, prompt):
    if not actual_api_key_to_use:
        return f"Error: No API key provided or resolved for provider '{provider}'. Cannot make API call."
    if not image_bytes:
        return "No image data provided for description."

    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    try:
        if provider == "openai":
            if not any(m_prefix in model for m_prefix in ["gpt-4", "gpt-3.5-turbo", "o1", "o3", "o4"]):
                 print(f"Warning: Selected OpenAI model '{model}' might not fully support vision or be optimized for it.")
            client = openai.OpenAI(api_key=actual_api_key_to_use)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=3000,
                timeout=120.0,
            )
            description = response.choices[0].message.content.strip()
            return description

        elif provider == "gemini":
             if not any(m_prefix in model for m_prefix in ["gemini-1.5", "gemini-pro-vision"]):
                 print(f"Warning: Selected Gemini model '{model}' might not optimally support vision. Prefer 'gemini-1.5-pro/flash' or 'gemini-pro-vision'.")
             genai.configure(api_key=actual_api_key_to_use)
             safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
             ]
             client = genai.GenerativeModel(model_name=model, safety_settings=safety_settings)
             image_part = {"mime_type": "image/jpeg", "data": image_bytes}
             response = client.generate_content([prompt, image_part], request_options={"timeout": 120})
             description = response.text.strip()
             return description

        elif provider == "groq":
             print(f"Note: Using Groq provider for model '{model}'. Ensure this model supports vision and the image_url parameter via Groq's OpenAI-compatible API. Text-only models will ignore the image.")
             client = Groq(api_key=actual_api_key_to_use)
             messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
             ]
             response = client.chat.completions.create(
                 model=model, messages=messages, max_tokens=3000, timeout=120.0
             )
             description = response.choices[0].message.content.strip()
             return description
        else:
            return f"Unknown LLM provider: {provider}"

    except requests.exceptions.Timeout:
         return f"LLM request timed out for {provider} (generic requests library timeout)."
    except openai.APITimeoutError:
        return f"OpenAI API request timed out."
    except genai.types.generation_types.StopCandidateException as e:
        reason = "Unknown"
        try:
            if e.args and e.args[0] and e.args[0].candidates and e.args[0].candidates[0].finish_reason:
                reason = e.args[0].candidates[0].finish_reason.name
        except (AttributeError, IndexError, TypeError): pass
        print(f"Gemini content generation stopped. Reason: {reason}. Full error: {e}")
        return f"Gemini: Content generation stopped. Prompt or response may have been blocked. Reason: {reason}"
    except Exception as e:
        if isinstance(e, openai.AuthenticationError):
            return f"OpenAI Authentication Error: Invalid API key or organization. ({e})"
        print(f"LLM API Error ({provider} - {model}): {e.__class__.__name__}: {e}")
        return f"Error generating description from {provider} ({model}): {e}. Check API key, model, image format, and ensure the model supports vision and your account has credit/access."

# --- Image Loading and Processing Helpers ---
def pil_to_bytes(image: Image.Image, format='JPEG', quality=90):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=format, quality=quality)
    return byte_arr.getvalue()

def calculate_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

# --- Optional API Key Input Node ---
class TenosaiAPIKeyInputNode:
    NODE_DISPLAY_NAME = "Tenosai API Key Input"
    CATEGORY = "utilities/TenosaiLLM"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key_string": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "get_key"

    def get_key(self, api_key_string):
        return (api_key_string.strip(),)

# --- ComfyUI Main Node Class ---
class TenosaiLLMImageDescribeNode:
    NODE_DISPLAY_NAME = "Tenosai LLM Image Describe"
    CATEGORY = "conditioning/TenosaiLLM"

    PROMPT_MODES = ["Style", "Subject", "Describe"]
    PROMPT_MAP = {
        "Style": STYLE_PROMPT,
        "Subject": SUBJECT_PROMPT,
        "Describe": DESCRIBE_PROMPT,
    }

    def __init__(self):
        self.last_image_hash = None
        self.cached_description = ""
        self.cache_settings_provider = None
        self.cache_settings_model = None
        self.cache_settings_mode = None

    @classmethod
    def INPUT_TYPES(s):
        provider_options = AVAILABLE_PROVIDERS if AVAILABLE_PROVIDERS else ["None"]
        model_options = AVAILABLE_MODELS_FLAT if AVAILABLE_MODELS_FLAT else ["None: Select a model"]

        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "llm_provider": (provider_options, {"default": provider_options[0]}),
                "llm_model": (model_options, {"default": model_options[0]}),
                "description_mode": (s.PROMPT_MODES, {"default": s.PROMPT_MODES[0]}),
                "auto_describe_on_image_change": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "override_api_key": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"

    def execute(self, clip, image, llm_provider, llm_model, description_mode, auto_describe_on_image_change, override_api_key=None):
        print(f"\n--- TenosaiLLMImageDescribeNode EXECUTE ---")
        print(f"Received - Provider: '{llm_provider}', Model: '{llm_model}', Mode: '{description_mode}', AutoDescribe: {auto_describe_on_image_change}, Override Key Provided: {override_api_key is not None and override_api_key.strip() != ''}")

        image_bytes = None
        current_image_hash = None
        image_processing_error = ""
        try:
            pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)).convert("RGB")
            image_bytes = pil_to_bytes(pil_image, format='JPEG')
            current_image_hash = calculate_image_hash(image_bytes)
        except Exception as e:
            image_processing_error = f"Error processing input image for LLM: {e}"
            print(image_processing_error)

        final_api_key_to_use = None
        if override_api_key and override_api_key.strip():
            final_api_key_to_use = override_api_key.strip()
            print("Using API key from 'override_api_key' input.")
        elif llm_provider != "None":
            api_key_name_from_map = API_KEY_MAP.get(llm_provider)
            if api_key_name_from_map and api_key_name_from_map in API_KEYS and API_KEYS[api_key_name_from_map]:
                final_api_key_to_use = API_KEYS[api_key_name_from_map]
                print(f"Using API key for '{llm_provider}' from api_keys.json.")
            else:
                print(f"No API key found for '{llm_provider}' in api_keys.json and no override provided.")
        else:
            print("LLM Provider is 'None', no API key resolution attempted.")

        image_changed = (current_image_hash != self.last_image_hash) if current_image_hash is not None else True
        current_model_name_only = llm_model.split(":", 1)[-1].strip() if ":" in llm_model else llm_model

        settings_changed_for_cache = (
            self.cache_settings_provider != llm_provider or
            self.cache_settings_model != current_model_name_only or
            self.cache_settings_mode != description_mode
        )

        if settings_changed_for_cache:
            print(f"Settings changed (Provider/Model/Mode). Cache for previous settings ('{self.cache_settings_provider}/{self.cache_settings_model}' in mode '{self.cache_settings_mode}') is now invalid for current settings ('{llm_provider}/{current_model_name_only}' in mode '{description_mode}').")
            self.cached_description = ""

        generated_description = ""

        llm_call_is_needed = (
            image_bytes is not None and
            final_api_key_to_use is not None and
            llm_provider != "None" and llm_model != "None: Select a model" and
            (
                (auto_describe_on_image_change and image_changed) or
                settings_changed_for_cache
            )
        )
        
        if llm_call_is_needed:
            print(f"LLM call triggered. Image changed: {image_changed}, Auto-describe: {auto_describe_on_image_change}, Settings (Prov/Model/Mode) changed: {settings_changed_for_cache}.")
            current_prompt_text = self.PROMPT_MAP.get(description_mode, DESCRIBE_PROMPT)
            
            print(f"Requesting description from {llm_provider}/{current_model_name_only} (mode: {description_mode})...")
            llm_output = get_llm_description(image_bytes, llm_provider, current_model_name_only, final_api_key_to_use, current_prompt_text)
            print(f"LLM Output ({description_mode} mode): {llm_output[:300]}{'...' if len(llm_output) > 300 else ''}")
            
            generated_description = llm_output
            self.cached_description = llm_output
            
            self.cache_settings_provider = llm_provider
            self.cache_settings_model = current_model_name_only
            self.cache_settings_mode = description_mode
            
            if current_image_hash is not None:
                self.last_image_hash = current_image_hash
        else:
            print("LLM call skipped. Reasons could include: no API key, image processing error, provider/model not selected, auto_describe is OFF and image/settings haven't changed necessitating a new call.")
            if settings_changed_for_cache:
                self.cache_settings_provider = llm_provider
                self.cache_settings_model = current_model_name_only
                self.cache_settings_mode = description_mode
            if image_changed and current_image_hash is not None:
                self.last_image_hash = current_image_hash

        description_for_clip = generated_description
        if not description_for_clip and self.cached_description and not settings_changed_for_cache:
            description_for_clip = self.cached_description
            print(f"Using cached description for '{llm_provider}/{current_model_name_only}' mode '{description_mode}'.")
        elif not description_for_clip and image_processing_error:
             description_for_clip = image_processing_error
             print("Using image processing error as description.")
        elif not description_for_clip and llm_call_is_needed and not generated_description:
             description_for_clip = f"LLM call was attempted for '{description_mode}' but failed to produce description."
        elif not description_for_clip:
             description_for_clip = ""
             print("No description generated or valid cache found; using empty string for CLIP.")
        
        # --- CLIP Encoding Logic ---
        conditioning = None
        encode_method = None
        method_name = "unknown method"

        print(f"CLIP object type: {type(clip).__name__}")

        # Priority 1: encode_from_tokens + tokenize (Standard ComfyUI approach for full output, includes pooled)
        if hasattr(clip, 'encode_from_tokens') and callable(clip.encode_from_tokens) and hasattr(clip, 'tokenize') and callable(clip.tokenize):
             encode_method = lambda text: clip.encode_from_tokens(clip.tokenize(text), return_pooled=True, return_dict=True)
             method_name = 'encode_from_tokens + tokenize (Standard/Pooled)'
             print("Using standard ComfyUI encode_from_tokens + tokenize.")
        # Priority 2: Standard encode_conditioning
        elif hasattr(clip, 'encode_conditioning') and callable(clip.encode_conditioning):
            encode_method = clip.encode_conditioning
            method_name = 'encode_conditioning (Standard)'
            print("Using standard encode_conditioning.")
        # Priority 3: Check for the 'encode' method
        elif hasattr(clip, 'encode') and callable(clip.encode):
             encode_method = clip.encode
             method_name = 'encode (Likely embeddings only)'
             print("Using 'encode' method - WARNING: May not include pooled_output for Flux/SDXL.")
        # Priority 4: Common custom/Flux encode_text
        elif hasattr(clip, 'encode_text') and callable(clip.encode_text):
             encode_method = clip.encode_text
             method_name = 'encode_text (Flux/Custom)'
             print("Using 'encode_text' method.")
        # Priority 5: Check if the object itself is callable
        elif callable(clip):
             encode_method = clip.__call__
             method_name = '__call__ (Callable CLIP)'
             print("Using callable CLIP object.")

        if encode_method is None:
            error_msg = f"Node Error: Connected CLIP object ({type(clip).__name__}) does not have a recognized text encoding method. Available methods: {dir(clip)}. Please connect a compatible CLIPLoader output."
            print(error_msg)
            # Fallback to empty conditioning to prevent crash
            try:
                target_dim = clip.get_target_dim() if hasattr(clip, 'get_target_dim') else 768
                text_feat_dim = target_dim
                if hasattr(clip, 'cond_stage_model') and hasattr(clip.cond_stage_model, 'transformer'):
                    text_feat_dim = clip.cond_stage_model.transformer.width
                empty_embeddings = torch.zeros((1, 77, text_feat_dim), device=clip.device if hasattr(clip, 'device') else 'cpu')
                empty_pooled = torch.zeros((1, target_dim), device=clip.device if hasattr(clip, 'device') else 'cpu')
                conditioning = [[empty_embeddings, {"pooled_output": empty_pooled}]]
            except Exception as fallback_e:
                print(f"Critical error creating fallback empty conditioning: {fallback_e}")
                conditioning = []
            return (conditioning,)


        text_to_encode = description_for_clip if description_for_clip and not description_for_clip.startswith("Error:") and not description_for_clip.startswith("LLM call was attempted") else ""
        if description_for_clip.startswith("Error:") or description_for_clip.startswith("LLM call was attempted"):
            print(f"Warning: Description for CLIP is an error/status message: '{description_for_clip}'. Encoding empty string instead.")

        try:
            print(f"Attempting CLIP encoding using method: '{method_name}' with text (first 200 chars): '{text_to_encode[:200]}{'...' if len(text_to_encode) > 200 else ''}'")
            encoded_output = encode_method(text_to_encode)
            print(f"Successfully executed encoding method '{method_name}'. Raw output type: {type(encoded_output)}")

            text_embeddings = None
            pooled_output = None
            conditioning_list_formatted = None

            if method_name == 'encode_from_tokens + tokenize (Standard/Pooled)' and isinstance(encoded_output, dict):
                 if 'cond' in encoded_output and isinstance(encoded_output['cond'], torch.Tensor):
                      text_embeddings = encoded_output['cond']
                      print("Extracted 'cond' (embeddings) from dictionary output.")
                      if 'pooled_output' in encoded_output and isinstance(encoded_output['pooled_output'], torch.Tensor):
                           pooled_output = encoded_output['pooled_output']
                           print("Extracted 'pooled_output' from dictionary output.")
                      else:
                           print("Warning: 'pooled_output' key not found or not a Tensor in dictionary output from encode_from_tokens.")
                 else:
                      raise ValueError(f"CLIP Encoding dict output from '{method_name}' missing 'cond' or not a Tensor. Keys: {encoded_output.keys()}.")
            elif isinstance(encoded_output, torch.Tensor):
                 text_embeddings = encoded_output
                 print("Encoded output recognized as a single Tensor (embeddings).")
            elif isinstance(encoded_output, (list, tuple)):
                 if len(encoded_output) >= 1 and isinstance(encoded_output[0], list) and len(encoded_output[0]) >= 1 and isinstance(encoded_output[0][0], torch.Tensor):
                      conditioning_list_formatted = encoded_output
                      print("Encoded output recognized as [[tensor, {}], ...] format.")
                 elif len(encoded_output) >= 2 and isinstance(encoded_output[0], torch.Tensor) and isinstance(encoded_output[1], torch.Tensor):
                      text_embeddings = encoded_output[0]
                      pooled_output = encoded_output[1]
                      print("Encoded output recognized as (embeddings, pooled_output) tuple/list.")
                 elif len(encoded_output) >= 1 and isinstance(encoded_output[0], torch.Tensor):
                      text_embeddings = encoded_output[0]
                      print("Encoded output recognized as a list/tuple containing at least one Tensor (using first as embeddings).")
                 else:
                     raise ValueError(f"CLIP Encoding list/tuple output unexpected format: {type(encoded_output)}. Content: {encoded_output}")
            else:
                 raise ValueError(f"CLIP Encoding produced unexpected output type: {type(encoded_output)}. Output: {encoded_output}")

            if text_embeddings is not None and conditioning_list_formatted is None:
                conditioning_item = [text_embeddings, {}]
                if pooled_output is not None:
                    conditioning_item[1]['pooled_output'] = pooled_output
                    print("Added 'pooled_output' to conditioning dictionary.")
                else:
                     if method_name == 'encode (Likely embeddings only)':
                         print(f"Warning: Method '{method_name}' returned only embeddings. May lack pooled_output for Flux/SDXL.")
                conditioning_list_formatted = [conditioning_item]
                print("Constructed final conditioning list from extracted data.")
            elif conditioning_list_formatted is not None:
                print("Using pre-formatted conditioning directly from CLIP output.")
            
            if conditioning_list_formatted is None:
                 raise ValueError("Failed to construct final conditioning list after processing encoded output.")
            
            conditioning = conditioning_list_formatted

        except Exception as e:
            print(f"Error during CLIP encoding/formatting with '{method_name}' on {type(clip).__name__}: {e}")
            try:
                print("Attempting fallback: encoding empty string for conditioning.")
                target_dim = clip.get_target_dim() if hasattr(clip, 'get_target_dim') else 768
                text_feat_dim = target_dim
                if hasattr(clip, 'cond_stage_model') and hasattr(clip.cond_stage_model, 'transformer'):
                    text_feat_dim = clip.cond_stage_model.transformer.width

                empty_embeddings = torch.zeros((1, 77, text_feat_dim), device=clip.device if hasattr(clip, 'device') else 'cpu')
                empty_pooled = torch.zeros((1, target_dim), device=clip.device if hasattr(clip, 'device') else 'cpu')
                conditioning = [[empty_embeddings, {"pooled_output": empty_pooled}]]
                print(f"Successfully created empty fallback conditioning. Dims emb: {text_feat_dim}, pooled: {target_dim}")
            except Exception as final_e:
                 print(f"Critical Error: Fallback empty conditioning creation failed: {final_e}")
                 conditioning = []
        
        if conditioning is None:
             conditioning = []
        
        print("--- TenosaiLLMImageDescribeNode EXECUTE END ---")
        return (conditioning,)


# --- Node Mapping ---
NODE_CLASS_MAPPINGS = {
    "TenosaiLLMImageDescribe": TenosaiLLMImageDescribeNode,
    "TenosaiAPIKeyInput": TenosaiAPIKeyInputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TenosaiLLMImageDescribe": "Tenosai LLM Image Describe",
    "TenosaiAPIKeyInput": "Tenosai API Key Input",
}