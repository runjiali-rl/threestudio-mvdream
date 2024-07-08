import argparse
from openai import OpenAI
import re
from diffusion_models import DiffusionModel
from partdream_recaptioning import image_prompt_iteration
import torch
import os
from tqdm import tqdm
import json



torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="custom/threestudio-mvdream/mllm_optimizer/intermediate_save")
    parser.add_argument("--cache_dir", type=str, default="/homes/55/runjia/scratch/diffusion_model_weights")
    parser.add_argument("--part_model_name", type=str, default="mvdream,deepfloyd")
    parser.add_argument("--global_model_name", type=str, default="mvdream,stable_diffusion_3")
    parser.add_argument("--iteration_num", type=int, default=3)
    parser.add_argument("--composite_description", type=str, default="A dog with a frog head and dragon wings")


    return parser.parse_args()


def initialize_prompt(composite_description,
                      api_key,
                      max_trial_times=100):
    initialize_meta_prompt_path = "custom/threestudio-mvdream/mllm_optimizer/prompts/initialization_prompt.txt"
    with open(initialize_meta_prompt_path, "r") as f:
        initialize_meta_prompt = f.read()

    initialize_meta_prompt = initialize_meta_prompt.replace("COMPOSITE_PROMPT", composite_description)
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
                            "content": initialize_meta_prompt,
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
        if "body" in part_name.lower(): # this is already captured in the composite creature description
            continue
        parts.append({
            'Part Name': part_name,
            'Description': part_description,
            'Spatial Center': spatial_center,
            'Radius': radius,
            'Shape': shape
        })
    
    return {
        'Parts': parts
    }



def main():
    args = parse_args()
    save_dir = args.save_dir
    cache_dir = args.cache_dir
    iteration_num = args.iteration_num
    composite_description = args.composite_description
    part_meta_prompt_path = "custom/threestudio-mvdream/mllm_optimizer/prompts/part_iterative_recaptioning_prompt.txt"
    global_meta_prompt_path = "custom/threestudio-mvdream/mllm_optimizer/prompts/global_iterative_recaptioning_prompt.txt"

    file_name = composite_description.replace(" ", "_")
    if os.path.exists(f"{save_dir}/saved_prompts/{file_name}_optimized_prompts.json"):
        return 

    structured_results = initialize_prompt(composite_description,
                                            args.api_key)
    
    part_model_names = args.part_model_name.split(",")
    global_model_names = args.global_model_name.split(",")

    part_models = [ DiffusionModel(part_model_name,
                                    cache_dir=cache_dir) for part_model_name in part_model_names]

    original_negative_prompt = "hollow space, gaps, ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions, uneven, uneven surface, stripes, disconnected, cartoon"
    optimized_prompts = {}
    optimized_negative_prompts = {}

    for part in structured_results["Parts"]:
        original_prompt = part["Description"]
        prompt_path = original_prompt.replace(" ", "_")
        prompt = original_prompt
        negative_prompt = original_negative_prompt
        for iteration in tqdm(range(iteration_num)):
            os.makedirs(f"{save_dir}/part/{prompt_path}/{iteration}", exist_ok=True)
            # Generate images for part models
            for part_model_idx, part_model in enumerate(part_models):
                images = part_model.generate_images(prompt, negative_prompt=negative_prompt)

                if isinstance(images, list):
                    for i, image in enumerate(images):
                        image.save(f"{save_dir}/part/{prompt_path}/{iteration}/{part_model_names[part_model_idx]}_{i}.png")
                else:
                    images.save(f"{save_dir}/part/{prompt_path}/{iteration}/{part_model_names[part_model_idx]}.png")
            
            image_paths = os.listdir(f"{save_dir}/part/{prompt_path}/{iteration}")
            image_paths = [f"{save_dir}/part/{prompt_path}/{iteration}/{image_path}" for image_path in image_paths]
            prompt, negative_prompt = image_prompt_iteration(part_meta_prompt_path,
                                                            original_prompt,
                                                            prompt,
                                                            negative_prompt,
                                                            image_paths,
                                                            args.api_key
                                                            )

        optimized_prompts[original_prompt] = prompt
        optimized_negative_prompts[original_prompt] = negative_prompt

    del part_models
    global_models = [ DiffusionModel(global_model_name,
                                    cache_dir=cache_dir) for global_model_name in global_model_names]
    original_prompt = composite_description
    prompt_path = original_prompt.replace(" ", "_")
    prompt = original_prompt
    negative_prompt = original_negative_prompt
    for iteration in tqdm(range(iteration_num)):
        os.makedirs(f"{save_dir}/global/{prompt_path}/{iteration}", exist_ok=True)
        # Generate images for global models
        for global_model_idx, global_model in enumerate(global_models):
            images = global_model.generate_images(prompt, negative_prompt=negative_prompt)

            if isinstance(images, list):
                for i, image in enumerate(images):
                    image.save(f"{save_dir}/global/{prompt_path}/{iteration}/{global_model_names[global_model_idx]}_{i}.png")
            else:
                images.save(f"{save_dir}/global/{prompt_path}/{iteration}/{global_model_names[global_model_idx]}.png")
        
        image_paths = os.listdir(f"{save_dir}/global/{prompt_path}/{iteration}")
        image_paths = [f"{save_dir}/global/{prompt_path}/{iteration}/{image_path}" for image_path in image_paths]
        prompt, negative_prompt = image_prompt_iteration(global_meta_prompt_path,
                                                        original_prompt,
                                                        prompt,
                                                        negative_prompt,
                                                        image_paths,
                                                        args.api_key
                                                        )
    
    optimized_global_prompt = prompt
    optimized_global_negative_prompt = negative_prompt
    optimized_prompts['global'] = optimized_global_prompt
    optimized_negative_prompts['global'] = optimized_global_negative_prompt

    os.makedirs(f"{save_dir}/saved_prompts", exist_ok=True)
    with open(f"{save_dir}/saved_prompts/{file_name}_optimized_prompts.json", "w") as f:
        json.dump(optimized_prompts, f, indent=4)
    with open(f"{save_dir}/saved_prompts/{file_name}_optimized_negative_prompts.json", "w") as f:
        json.dump(optimized_negative_prompts, f, indent=4)
    
    print(optimized_global_prompt)
    stop = 1










if __name__ == "__main__":
    main()