import argparse
from openai import OpenAI
import re
from diffusion_models import DiffusionModel
from partdream_recaptioning import image_prompt_iteration
import torch
import os
from tqdm import tqdm



torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="custom/threestudio-mvdream/intermediate_images")
    parser.add_argument("--cache_dir", type=str, default="/homes/55/runjia/storage/diffusion_model_weights")
    parser.add_argument("--part_model_name", type=str, default="mvdream,deepfloyd")
    parser.add_argument("--global_model_name", type=str, default="mvdream,stable_diffusion_3")
    parser.add_argument("--iteration_num", type=int, default=4)
    parser.add_argument("--composite_description", type=str, default="A dog with a cat head and dragon wings")
    parser.add_argument("--meta_prompt_path", type=str, default="custom/threestudio-mvdream/prompt_processor/prompts/iterative_recaptioning_prompt.txt")

    return parser.parse_args()


def initialize_prompt(composite_description,
                      api_key,
                      max_trial_times=100):
    meta_prompt_path = "custom/threestudio-mvdream/prompt_processor/prompts/initialization_prompt.txt"
    with open(meta_prompt_path, "r") as f:
        meta_prompt = f.read()

    meta_prompt = meta_prompt.replace("COMPOSITE_PROMPT", composite_description)
    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
    )
    i = 0
    while i < max_trial_times:
        i += 1
        try:
            chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": meta_prompt,
                        }
                    ],
                    model="gpt-4o",
                )

            raw_result = chat_completion.choices[0].message.content

            structured_result = parse_composite_creature_description(raw_result)
            break
        except:
            structured_result = None
            continue 

    if not structured_result:
        raise Exception("the input prompt cannot be parsed, please try another composite prompt")
    
    return structured_result




def parse_composite_creature_description(description: str) -> dict:
    # Define regex patterns to capture the different parts of the prompt
    part_pattern = re.compile(r'\d+\.\s\*\*([a-zA-Z\s]+)\*\*\n\s+-\sDescription:\s([a-zA-Z\s]+)\n\s+-\sSpatial\sCenter:\s\(([-0-9.,\s]+)\)\n\s+-\sRadius:\s([0-9.]+)\n\s+-\sShape:\s([a-zA-Z]+)')
    
    # Find all matches
    matches = part_pattern.findall(description)
    
    # Parse the matches into structured data
    parts = []
    for match in matches:
        part_name = match[0].strip()
        part_description = match[1].strip()
        spatial_center = tuple(map(float, match[2].split(',')))
        radius = float(match[3])
        shape = match[4].strip()
        
        parts.append({
            'Part Name': part_name,
            'Description': part_description,
            'Spatial Center': spatial_center,
            'Radius': radius,
            'Shape': shape
        })
    
    return {
        'Composite Creature Description': description.split('\n')[0].split('**')[1].strip(),
        'Parts': parts
    }



def main():
    args = parse_args()
    save_dir = args.save_dir
    cache_dir = args.cache_dir
    iteration_num = args.iteration_num
    composite_description = args.composite_description
    meta_prompt_path = args.meta_prompt_path

    structured_results = initialize_prompt(composite_description,
                                            args.api_key)
    
    part_model_names = args.part_model_name.split(",")
    global_model_names = args.global_model_name.split(",")
    negative_prompt = "hollow space, gaps, ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions, uneven, uneven surface, stripes, disconnected, cartoon"


    for part in structured_results["Parts"]:
        original_prompt = part["Description"]
        prompt_path = original_prompt.replace(" ", "_")
        prompt = original_prompt
        for iteration in tqdm(range(iteration_num)):
            os.makedirs(f"{save_dir}/part/{prompt_path}/{iteration}", exist_ok=True)
            # Generate images for part models
            for part_model_name in part_model_names:
                model = DiffusionModel(part_model_name,
                                    cache_dir=cache_dir)
                images = model.generate_images(prompt, negative_prompt=negative_prompt)

                if isinstance(images, list):
                    for i, image in enumerate(images):
                        image.save(f"{save_dir}/part/{prompt_path}/{iteration}/{part_model_name}_{i}.png")
                else:
                    images.save(f"{save_dir}/part/{prompt_path}/{iteration}/{part_model_name}.png")
            
            image_paths = os.listdir(f"{save_dir}/part/{prompt_path}/{iteration}")
            image_paths = [f"{save_dir}/part/{prompt_path}/{iteration}/{image_path}" for image_path in image_paths]
            prompt, new_negative_prompt = image_prompt_iteration(meta_prompt_path,
                                                            original_prompt,
                                                            prompt,
                                                            negative_prompt,
                                                            image_paths,
                                                            args.api_key
                                                            )
            negative_prompt = new_negative_prompt + ", " + negative_prompt
    
    print(prompt)
    stop = 1










if __name__ == "__main__":
    main()