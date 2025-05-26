from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

def load_model_and_processor():
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    
    # Load model in 8-bit
    model = AutoModelForImageTextToText.from_pretrained(
        model_checkpoint,
        device_map=torch_device,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16
    )
    return model, processor

def generate_description(model, processor, image_url):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
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
    model, processor = load_model_and_processor()
    
    # Example image URL
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    # Generate description
    description = generate_description(model, processor, image_url)
    print("Generated description:")
    print(description)

if __name__ == "__main__":
    main()