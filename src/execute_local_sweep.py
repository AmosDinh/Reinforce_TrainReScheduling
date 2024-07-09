import argparse
import yaml
import itertools
import subprocess
import sys

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_parameter_combinations(parameters):
    param_names = list(parameters.keys())
    param_values = [parameters[name]['values'] for name in param_names]
    return [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

def run_program(program, parameters):
    cmd = [sys.executable, program]
    for key, value in parameters.items():
        cmd.extend([f"--{key}", str(value)])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {program} with parameters {parameters}")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run grid search with YAML configuration")
    parser.add_argument("config", help="Path to the YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)
    program = config['program']
    parameters = config['parameters']

    combinations = generate_parameter_combinations(parameters)
    total_combinations = len(combinations)

    print(f"Total number param combis: {total_combinations}")

    for i, params in enumerate(combinations, 1):
        print(f"\nRunning combi {i}/{total_combinations}")
        print("Params:", params)
        run_program(program, params)

if __name__ == "__main__":
    main()