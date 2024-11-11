import argparse
import os
from tqdm import tqdm

def main(args):
    videos_path = args.video_folder
    # Walk all the videos in the path
    dir_list = sorted(os.listdir(videos_path))
    print("Writing annotations to file {}".format(args.output_file))
    with open(args.output_file, 'w') as f:
        for idx, dir_name in enumerate(dir_list):
            dir_path = os.path.join(videos_path, dir_name)
            for file_name in sorted(os.listdir(dir_path)):
                full_path = os.path.join(dir_path, file_name)
                f.write(f'{full_path} {idx}\n')
    print("Done! Validating...")
    # Validate the file.
    num_lines = sum(1 for _ in open(args.output_file, 'r'))
    for line in tqdm(open(args.output_file, 'r'), colour='green', total=num_lines):
        path, _ = line.strip().split()
        assert os.path.exists(path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    main(args)