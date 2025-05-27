# Building the MatrixToDiscourseBot Plugin

This document explains how to build the MatrixToDiscourseBot maubot plugin into a `.mbp` file that can be uploaded to your maubot instance.

## Prerequisites

- Python 3.8 or higher
- PyYAML package (`pip install pyyaml`)

## Build Methods

### Method 1: Using the Python Build Script (Recommended)

The easiest way to build the plugin is using the included Python script:

```bash
# From the repository root directory
python3 build_plugin.py
```

Options:
- `-v, --verbose`: Show detailed plugin metadata during build
- `-o, --output <filename>`: Specify a custom output filename (default: auto-generated from plugin ID and version)

Example:
```bash
python3 build_plugin.py -v -o my-custom-plugin.mbp
```

### Method 2: Using maubot CLI (mbc)

If you have the maubot CLI installed:

```bash
# Install mbc if not already installed
pip install maubot

# Run the build script
./build_plugin.sh
```

Or manually:
```bash
cd plugin
mbc build
mv *.mbp ../
```

### Method 3: Manual ZIP Creation

You can also create the `.mbp` file manually:

```bash
cd plugin
zip -9r ../matrix-to-discourse.mbp maubot.yaml *.py base-config.yaml
```

## Output

The build process will create a `.mbp` file in the repository root directory. The filename will be in the format:
- `{plugin_id}-v{version}.mbp` (e.g., `may.irregularchat.matrix_to_discourse-v1.0.1.mbp`)

## Uploading to Maubot

1. Log in to your maubot instance web interface
2. Go to the "Plugins" section
3. Click "Upload" and select the generated `.mbp` file
4. Create a new instance of the plugin or update existing instances

## Troubleshooting

- **"maubot.yaml not found"**: Make sure you're running the build script from the repository root directory, not from inside the `plugin/` directory
- **Missing files**: Ensure all Python modules listed in `maubot.yaml` exist in the `plugin/` directory
- **ZIP errors**: Make sure you have write permissions in the current directory 