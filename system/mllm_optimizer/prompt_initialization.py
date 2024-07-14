import argparse
from openai import OpenAI
import re
from .diffusion_models import DiffusionModel
from .partdream_recaptioning import image_prompt_iteration
import torch
import os
from tqdm import tqdm
import json



torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="custom/threestudio-mvdream/system/mllm_optimizer/intermediate_save")
    parser.add_argument("--cache_dir", type=str, default="/homes/55/runjia/scratch/diffusion_model_weights")
    parser.add_argument("--part_model_name", type=str, default="mvdream,deepfloyd")
    parser.add_argument("--global_model_name", type=str, default="mvdream,stable_diffusion_3")
    parser.add_argument("--iteration_num", type=int, default=3)
    parser.add_argument("--composite_description", type=str, default="A dog with a frog head and dragon wings")


    return parser.parse_args()


def initialize_prompt(composite_description,
                      api_key,
                      max_trial_times=100):
    initialize_meta_prompt_path = "custom/threestudio-mvdream/system/mllm_optimizer/prompts/initialization_prompt.txt"
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


def run_model_optimization(
    composite_description,
    part_model_names,
    global_model_names,
    api_key,
    iteration_num=3,
    cache_dir=None,
    save_dir= "custom/threestudio-mvdream/system/mllm_optimizer/intermediate_save"
):
    part_meta_prompt_path = "custom/threestudio-mvdream/system/mllm_optimizer/prompts/part_iterative_recaptioning_prompt.txt"
    global_meta_prompt_path = "custom/threestudio-mvdream/system/mllm_optimizer/prompts/global_iterative_recaptioning_prompt.txt"

    file_name = composite_description.replace(" ", "_")
    save_path_check = f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_part_prompts.json"
    if os.path.exists(save_path_check):
        print("The optimized prompts for this composite creature description have already been generated.")
        optimized_part_prompts = json.load(open(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_part_prompts.json"))
        optimized_negative_part_prompts = json.load(open(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_negative_part_prompts.json"))
        optimized_global_prompts = json.load(open(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_global_prompts.json"))
        optimized_negative_global_prompts = json.load(open(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_negative_global_prompts.json"))
        return optimized_global_prompts, optimized_negative_global_prompts, optimized_part_prompts, optimized_negative_part_prompts

    structured_results = initialize_prompt(composite_description, api_key)
    
    part_model_names = part_model_names.split(",")
    global_model_names = global_model_names.split(",")

    original_negative_prompt = (
        "hollow space, gaps, ugly, bad anatomy, blurry, pixelated, obscure, "
        "unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, "
        "artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, "
        "bad proportions, uneven, uneven surface, stripes, disconnected, cartoon"
    )



    optimized_part_prompts, optimized_negative_part_prompts = initialize_optimized_prompts(part_model_names)
    optimized_global_prompts, optimized_negative_global_prompts = initialize_optimized_prompts(global_model_names)


    process_models(
        part_model_names, part_meta_prompt_path, structured_results["Parts"], iteration_num,
        save_dir, cache_dir, original_negative_prompt, optimized_part_prompts, optimized_negative_part_prompts, api_key
    )

    process_models(
        global_model_names, global_meta_prompt_path, [{"Description": composite_description}], iteration_num,
        save_dir, cache_dir, original_negative_prompt, optimized_global_prompts, optimized_negative_global_prompts,
        api_key, global_mode=True
    )

    save_all_prompts(
        save_dir, file_name, optimized_part_prompts, optimized_negative_part_prompts,
        optimized_global_prompts, optimized_negative_global_prompts, iteration_num
    )

    return optimized_global_prompts, optimized_negative_global_prompts, optimized_part_prompts, optimized_negative_part_prompts


def initialize_optimized_prompts(model_names):
    return ({name: {} for name in model_names}, {name: {} for name in model_names})


def process_models(
    model_names, meta_prompt_path, structured_parts, iteration_num,
    save_dir, cache_dir, original_negative_prompt, optimized_prompts, optimized_negative_prompts,
    api_key, global_mode=False
):  
    if iteration_num == 0:
        models = [None for _ in model_names]
    else:
        models = [DiffusionModel(name, cache_dir=cache_dir) for name in model_names]

    for model_idx, model in enumerate(models):
        for part in structured_parts:
            original_prompt = part["Description"]
            prompt, negative_prompt = original_prompt, original_negative_prompt
            file_name = original_prompt.replace(" ", "_")

            for iteration in tqdm(range(iteration_num)):
                model_type = "global" if global_mode else "part"
                save_path = f"{save_dir}/{model_type}/{model_names[model_idx]}/{file_name}/{iteration}"
                os.makedirs(save_path, exist_ok=True)

                images = model.generate_images(prompt, negative_prompt=negative_prompt)
                save_images(images, save_path, model_type)

                image_paths = [os.path.join(save_path, img) for img in os.listdir(save_path)]
                prompt, negative_prompt = image_prompt_iteration(
                    meta_prompt_path, original_prompt, prompt, negative_prompt, image_paths, api_key
                )

            key = 'global' if global_mode else original_prompt
            optimized_prompts[model_names[model_idx]][key] = prompt
            optimized_negative_prompts[model_names[model_idx]][key] = negative_prompt

    del models


def save_images(images, save_path, prefix):
    if isinstance(images, list):
        for i, image in enumerate(images):
            image.save(os.path.join(save_path, f"{prefix}_{i}.png"))
    else:
        images.save(os.path.join(save_path, f"{prefix}.png"))


def save_all_prompts(save_dir,
                     file_name,
                     part_prompts,
                     negative_part_prompts,
                     global_prompts,
                     negative_global_prompts,
                     iteration_num=3):
    os.makedirs(f"{save_dir}/saved_prompts", exist_ok=True)
    save_prompts(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_part_prompts.json", part_prompts)
    save_prompts(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_negative_part_prompts.json", negative_part_prompts)
    save_prompts(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_global_prompts.json", global_prompts)
    save_prompts(f"{save_dir}/saved_prompts/{file_name}_{iteration_num}_optimized_negative_global_prompts.json", negative_global_prompts)


def save_prompts(path, prompts):
    with open(path, "w") as f:
        json.dump(prompts, f, indent=4)



def main():
    args = parse_args()
    run_model_optimization(
        args.composite_description,
        args.part_model_name,
        args.global_model_name,
        args.api_key,
        save_dir=args.save_dir,
        cache_dir=args.cache_dir,
        iteration_num=args.iteration_num,
    )






if __name__ == "__main__":
    main()