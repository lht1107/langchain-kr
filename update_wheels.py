import os
import glob
import subprocess
import sys
from packaging import version
from packaging.utils import parse_wheel_filename
import importlib.metadata

def get_installed_packages():
    """
    Retrieve a dictionary of installed packages and their versions.
    """
    return {dist.metadata['Name'].lower(): dist.version for dist in importlib.metadata.distributions()}

def parse_wheel_filename_custom(wheel_path):
    """
    Extract package name and version from the wheel filename using packaging.utils.
    """
    filename = os.path.basename(wheel_path)
    try:
        name, ver, _, _ = parse_wheel_filename(filename)
        return name.lower(), ver
    except Exception as e:
        print(f"Unable to parse wheel filename: {filename}. Error: {e}")
        return None, None

def identify_package_actions(installed_packages, wheels_dir):
    """
    Identify packages that need to be updated or installed based on the wheel files.
    Explicitly handles both upgrades and downgrades.
    """
    to_update = []
    to_install = []
    wheel_files = glob.glob(os.path.join(wheels_dir, "*.whl"))
    
    for wheel in wheel_files:
        name, ver = parse_wheel_filename_custom(wheel)
        if name:
            if name in installed_packages:
                installed_ver = version.parse(installed_packages[name])
                wheel_ver = version.parse(ver)
                if wheel_ver < installed_ver:  # Downgrade
                    print(f"Downgrading {name} from {installed_ver} to {wheel_ver}")
                    to_update.append((name, installed_packages[name], ver, wheel))
                elif wheel_ver > installed_ver:  # Upgrade
                    print(f"Upgrading {name} from {installed_ver} to {wheel_ver}")
                    to_update.append((name, installed_packages[name], ver, wheel))
            else:
                to_install.append((name, ver, wheel))
    
    return to_update, to_install

def remove_outdated_packages(to_update):
    """
    Uninstall outdated packages before updating them.
    """
    for name, old_version, _, _ in to_update:
        print(f"Removing outdated package: {name}=={old_version}")
        try:
            subprocess.check_call(['poetry', 'run', 'pip', 'uninstall', '-y', name])
            print(f"Successfully removed: {name}=={old_version}")
        except subprocess.CalledProcessError as e:
            print(f"Error uninstalling {name}: {e}")

def install_or_update_package(package_info):
    """
    Install or update a package using the provided wheel file.
    """
    name, ver, wheel_path = package_info
    print(f"Installing/Updating package: {name}=={ver}")
    try:
        subprocess.check_call(['poetry', 'run', 'pip', 'install', '--no-index', '--find-links=./wheels', wheel_path])
        print(f"Successfully installed: {name}=={ver}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {name}=={ver}: {e}")

def main():
    """
    Main function to manage the update and installation process.
    """
    wheels_dir = './wheels'
    
    if not os.path.isdir(wheels_dir):
        print(f"Wheels directory does not exist: {wheels_dir}")
        sys.exit(1)
    
    installed_packages = get_installed_packages()
    to_update, to_install = identify_package_actions(installed_packages, wheels_dir)
    
    print("Packages to update:")
    for name, old_ver, new_ver, _ in to_update:
        print(f"  - {name}: {old_ver} -> {new_ver}")
    
    print("New packages to install:")
    for name, ver, _ in to_install:
        print(f"  - {name}=={ver}")
    
    if to_update:
        remove_outdated_packages(to_update)
    
    all_actions = [(name, new_ver, wheel) for name, _, new_ver, wheel in to_update] + to_install
    
    for package_info in all_actions:
        install_or_update_package(package_info)

if __name__ == '__main__':
    main()
