import os
import json
import argparse

if __name__ == "__main__":

    classes = {"cat": 0, "dog": 1}
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='',
                         help="Path to directory with images")
    parser.add_argument("--out_json", type=str, default='',
                        help="Path to output json file")
    args = parser.parse_args()

    dataset = []
    for label in os.listdir(args.data_path):
        class_id = classes[label.lower()]
        img_dir = os.path.join(args.data_path, label)
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            dataset.append({"img_path": img_path, "label": class_id})

    with open(args.out_json, 'w') as outfile:
        json.dump(dataset, outfile, indent=2)
