# Demonstration Files Setup Guide

## Overview

The demonstration files (large Excel files, videos, etc.) are excluded from the main repository to keep it lightweight. This guide explains how to set up the demonstration files.

## Why Excluded?

- **Large file sizes**: Some demo files are 30-40MB
- **Repository performance**: Faster clones and pulls
- **Storage efficiency**: Reduces repository bloat
- **Flexibility**: Can be updated independently

## Required Files

You need the following files in your `demonstration/` folder:

```
demonstration/
├── CP_files/                                    # Control plan demo files
├── target_files/                                # Target file demos
├── graph_files/                                 # Graph/drawing demos
├── APQP_files/                                  # APQP stage demos
├── generated_files/                             # Pre-generated outputs
│   ├── prompt_output.txt
│   └── 2_symbol_check_result.txt
├── 副本LL-lesson learn-历史问题规避-V9.4.xlsx    # History issues Excel (41MB)
└── 大语言模型APQP_20250620.pptx                # APQP presentation (953KB)
```

## Setup Options

### Option 1: Download from Release Assets (Recommended)

1. Go to the [GitHub Release v1.0.0](https://github.com/Wa/pqm-ai-quality-control/releases/tag/v1.0.0)
2. Download `demo-files.zip`
3. Extract to your project root directory
4. Verify the `demonstration/` folder is created

**Direct Download Link**: [demo-files.zip](https://github.com/Wa/pqm-ai-quality-control/releases/download/v1.0.0/demo-files.zip)

### Option 2: Manual Setup

1. Create the `demonstration/` folder structure:
   ```bash
   mkdir -p demonstration/{CP_files,target_files,graph_files,APQP_files,generated_files}
   ```

2. Add your demo files to the appropriate folders

3. Create sample files for testing:
   ```bash
   # Create sample control plan files
   touch demonstration/CP_files/sample_control_plan.xlsx
   
   # Create sample target files
   touch demonstration/target_files/sample_target.xlsx
   
   # Create sample generated outputs
   echo "Sample prompt output" > demonstration/generated_files/prompt_output.txt
   echo "Sample check result" > demonstration/generated_files/2_symbol_check_result.txt
   ```

### Option 3: Use Git LFS (Advanced)

If you want to track large files properly:

```bash
# Install Git LFS
sudo apt install git-lfs

# Initialize LFS
git lfs install

# Track large files
git lfs track "demonstration/*.xlsx"
git lfs track "demonstration/*.pptx"
git lfs track "media/*.webm"

# Add tracking rules
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Now you can add large files normally
git add demonstration/
git commit -m "Add demonstration files with LFS"
```

## Verification

After setup, verify the demonstration works:

1. Run the application: `streamlit run main.py`
2. Go to the "特殊特性符号检查" tab
3. Click "演示" button
4. Verify demo files are loaded and analysis works

## Troubleshooting

### "Demo files not found" Error
- Ensure `demonstration/` folder exists in project root
- Check that required subfolders are present
- Verify file permissions

### Large File Upload Issues
- Use Git LFS for files > 50MB
- Consider splitting large files
- Use external storage for very large files

### Performance Issues
- Large demo files may slow down the application
- Consider using smaller sample files for development
- Use full demo files only for testing

## Best Practices

1. **Keep demo files separate** from main code
2. **Document file requirements** clearly
3. **Provide multiple setup options** for different users
4. **Version control demo files** separately if they change frequently
5. **Use appropriate file formats** (Excel for data, images for visuals)

## File Size Guidelines

- **< 1MB**: Include in main repository
- **1-50MB**: Use Git LFS or release assets
- **> 50MB**: Use external storage or split files 