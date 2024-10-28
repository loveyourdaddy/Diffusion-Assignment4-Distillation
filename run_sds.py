import json
import subprocess
import sys

def run_prompts():
    with open('./data/prompt_img_pairs.json', 'r') as file:
        config = json.load(file)

    # 각 프롬프트에 대해 처리
    for key, value in config.items():
        prompt = value['prompt']
        print(f"\nProcessing prompt: {prompt}")
        
        # Python 스크립트 실행
        command = f'python main.py --prompt "{prompt}" --loss_type sds --guidance_scale 25'
        print(command)
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for prompt '{prompt}': {e}")
            continue

if __name__ == "__main__":
    run_prompts()