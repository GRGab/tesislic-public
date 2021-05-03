import subprocess

from utils import save_to_temp

def calculate_I_complexity(binary_string, tempfile='temp/temp'):
    save_to_temp(binary_string, 'text', tempfile=tempfile)
    try:
        completed_process = subprocess.run(['./independence_tests/I_complexity.out',
                                            tempfile, '2'],
                                            capture_output=True,
                                            check=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise
    I = float(completed_process.stdout)
    return I