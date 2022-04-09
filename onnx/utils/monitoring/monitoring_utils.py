"""Small script that automates copying of monitoring app files."""
import shutil


def copy_monitoring_app(path):
    """Copy the monitoring Flask app and its related files to path."""
    shutil.copy(
        'utils/monitoring/monitoring.py',
        path + '/monitoring.py'
    )
    shutil.copy(
        'utils/monitoring/Dockerfile',
        path + '/Dockerfile'
    )
    shutil.copy(
        'utils/monitoring/requirements.txt',
        path + '/requirements.txt'
    )


if __name__ == '__main__':
    pass
