from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from PIL import Image
import torch

def load_model_and_processor():
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_checkpoint,
        trust_remote_code=True
    )
    # Load config (as user did, ensuring trust_remote_code)
    config = AutoConfig.from_pretrained(
        model_checkpoint,
        trust_remote_code=True
    )
    
    # Load model in 8-bit
    model = AutoModelForImageTextToText.from_pretrained(
        model_checkpoint,
        # config=config, # Can pass config explicitly if desired
        device_map=torch_device,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    return model, processor, config

def get_projector_output(model, pixel_values):
    """
    Extracts the output from the MLP projector of the InternVL model
    by using the model's `get_image_features` method, based on error feedback.
    Args:
        model: The loaded InternVL model (potentially wrapped).
        pixel_values: Processed pixel values of the image.
    Returns:
        Output tensor from the MLP projector.
    """
    # Determine the core model, in case of quantization wrappers like BitsAndBytes
    core_model = model
    if hasattr(model, 'model') and hasattr(model.model, 'get_image_features'):
        core_model = model.model
    elif not hasattr(core_model, 'get_image_features'):
        # If neither model nor model.model has get_image_features, raise an error.
        attrs_model = [attr for attr in dir(model) if 'image' in attr.lower() or 'feature' in attr.lower()]
        attrs_core_model = []
        if hasattr(model, 'model'):
             attrs_core_model = [f'model.{attr}' for attr in dir(model.model) if 'image' in attr.lower() or 'feature' in attr.lower()]
        all_relevant_attrs = list(set(attrs_model + attrs_core_model))
        raise AttributeError(f"Model or model.model does not have a 'get_image_features' method. Relevant attributes found: {all_relevant_attrs}. Cannot reliably get projector output.")

    # Ensure pixel_values are on the correct device and dtype for the core_model's method
    target_dtype = torch.bfloat16 # Default
    # Try to get dtype from core_model, otherwise keep bfloat16
    if hasattr(core_model, 'dtype') and core_model.dtype is not None:
        target_dtype = core_model.dtype
    elif hasattr(core_model, 'config') and hasattr(core_model.config, 'torch_dtype') and core_model.config.torch_dtype is not None:
        target_dtype = core_model.config.torch_dtype
    
    # The device should be the device of the core_model or its parameters
    try:
        target_device = next(core_model.parameters()).device
    except StopIteration: # Model might have no parameters directly (e.g. if not yet fully built or empty)
        target_device = model.device # Fallback to the wrapper's device

    pixel_values = pixel_values.to(device=target_device, dtype=target_dtype)

    # Use the model's own get_image_features method.
    # This method typically returns the features after they have passed through the MLP projector.
    
    # Get vision_feature_layer and vision_feature_select_strategy from config
    # Default values are based on the printed config if not found, though they should be there.
    vision_feature_layer = getattr(core_model.config, 'vision_feature_layer', -1)
    vision_feature_select_strategy = getattr(core_model.config, 'vision_feature_select_strategy', 'default')

    output = core_model.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy
    )
    
    # It might return a tuple (e.g., image_embeds, image_attention_mask), so we take the first element.
    if isinstance(output, tuple):
        projected_features = output[0]
    else:
        projected_features = output
        
    return projected_features


def generate_description(model, processor, image_path):
    from PIL import Image
    
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Please describe the image explicitly."},
            ],
        }
    ]

    # Process input
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    # Generate description
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    decoded_output = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return decoded_output

def main():
    # Load model and processor
    model, processor, config = load_model_and_processor()
    
    print("Model Configuration:")
    print(config)
    print("-"*50) 
    
    # Local image path
    image_path = "./test_images/1.jpg"

    # Prepare image for projector output extraction
    pil_image = Image.open(image_path).convert('RGB')
    
    # Process image using the image_processor component of the AutoProcessor
    if hasattr(processor, 'image_processor') and processor.image_processor is not None:
        image_inputs = processor.image_processor(images=pil_image, return_tensors="pt")
    elif hasattr(processor, 'feature_extractor') and processor.feature_extractor is not None: # Fallback for older naming
        image_inputs = processor.feature_extractor(images=pil_image, return_tensors="pt")
    else:
        raise AttributeError("Processor does not have 'image_processor' or 'feature_extractor'. Cannot process image for vision model.")

    pixel_values = image_inputs['pixel_values']

    # Get and print MLP projector output
    projector_output = get_projector_output(model, pixel_values)
    print("\nMLP Projector Output Shape:", projector_output.shape)
    print("MLP Projector Output (first image, first token, first 5 features):")
    if projector_output.numel() > 0: # Check if tensor is not empty
        print(projector_output[0, 0, :5]) 
    else:
        print("Projector output is empty.")
    print("-"*50)

    # Generate description (original functionality)
    print("\nAttempting to generate description...")
    description = generate_description(model, processor, image_path)
    print("Generated description:")
    print(description)

if __name__ == "__main__":
    main()