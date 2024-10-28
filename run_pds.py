import json
import subprocess
import sys

def run_prompts():
    with open('./data/prompt_img_pairs.json', 'r') as file:
        config = json.load(file)

    # 각 프롬프트에 대해 처리
    for key, value in config.items():
        prompt = value['prompt']
        edit_prompt = value['edit_prompt']
        img_path = value['img_path']
        print(f"\nProcessing prompt: {prompt}")
        print(f"edit_prompt: {edit_prompt}")

        # img_path.repla`
        img_path = img_path.replace('$HOME', '.')
        print(f"img_path: {img_path}")
        
        # Python 스크립트 실행
        # command = f'python main.py --prompt "{prompt}" --loss_type sds --guidance_scale 25'
        command = f'python main.py --prompt "{prompt}" --edit_prompt "{edit_prompt}" --src_img_path "{img_path}" --loss_type pds --guidance_scale 7.5 --step 200'

        print(command)
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for prompt '{prompt}': {e}")
            continue

if __name__ == "__main__":
    run_prompts()