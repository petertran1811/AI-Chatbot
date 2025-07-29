import yaml
import os


def load_config():
    with open("src/configs/models.yaml") as stream:
        return yaml.safe_load(stream)


def is_directory(resource_name):
    return '.' not in resource_name


def build_download_prefix_list(master_config):
    prefix_list = []
    precision = master_config['global']['precision']
    cuda_devices = master_config['global'].get('cuda_devices', [])

    for model_name, model in master_config.items():
        if model_name == 'global' or "resources" not in model:
            continue

        resources = model.get("resources", {})

        local_precision = model.get("precision", precision)
        model_id = resources.get('model_name', None)

        if model_id is None:
            prefix_list.append({
                'model': ".",
                'prefix': resources['checkpoint'],
                'is_dir': True
            })

        else:
            for key, value in resources.items():
                if key not in ["model_name"]:
                    if key == "engine" and value == "trt":
                        for cuda_device in cuda_devices:
                            prefix_list.append({
                                'model': model_id,
                                'prefix': f"{model_id}.{local_precision}.trt.{cuda_device}",
                                'is_dir': False
                            })
                    else:
                        prefix_list.append({
                            'model': model_id,
                            'prefix': value,
                            'is_dir': False
                        })

    return prefix_list


def download_from_s3(master_config, prefix_list):
    aws_profile = master_config['global']['s3']['aws_profile']
    bucket_name = master_config['global']['s3']['bucket_name']

    for download_prefix in prefix_list:
        print(aws_profile, bucket_name, download_prefix['model'], download_prefix['prefix'], str(
            download_prefix['is_dir']))


def main():
    master_config = load_config()
    prefix_list = build_download_prefix_list(master_config)
    download_from_s3(master_config, prefix_list)


if __name__ == "__main__":
    main()