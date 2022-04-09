"""Small script that automates copying of monitoring app files."""
import shutil
import os
import yaml


def init_folder_structure(
        build_path,
        monitoring_params=None
):
    """Create an empty build folder tree at path."""
    build_path_inference = build_path + '/inference'
    build_path_monitoring = build_path + '/monitoring'
    if monitoring_params is not None:
        shutil.copytree(
            'template',
            build_path
        )
        shutil.copy(
            monitoring_params['reference_set_path'],
            build_path_monitoring + '/references.parquet'
        )
        shutil.copy(
            monitoring_params['grafana_dashboard_path'],
            build_path + '/grafana/provisioning/dashboards/dashboard.json'
        )
        with open(build_path_monitoring + '/evidently.yaml', 'w')\
             as evidently_config_file:
            yaml.dump(
                monitoring_params['evidently_config'],
                evidently_config_file
            )
    if not os.path.exists(build_path_inference):
        os.makedirs(build_path_inference)
    return build_path_inference


if __name__ == '__main__':
    pass
