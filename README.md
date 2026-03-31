# A Two-Stage Conditional Diffusion Model for Underwater Image Enhancement (TCDM-UIE)  
The initial version of the code is uploaded. Further updates to follow.  
  
## Requirements 
python>=3.9   
torch>=2.0  
torchvision  
numpy  
pandas   
opencv-python     
skimage   
scipy    
tqdm  
lmdb    
pillow    
yaml     
einops    
lpips  
tensorboard  
wandb    

## datasets  
├── UIEB/  
│   ├── train/  
│   │   ├── input/  
│   │   │   ├── 0001.png/  
│   │   │   ├── ...  
│   │   │   └── 0800.png/  
│   │   ├── gt/  
│   │   │   ├── 0001.png/  
│   │   │   ├── ...  
│   │   │   └── 0800.png/  


## Acknowledgments   
Part of the code is adapted from previous works: [Restormer](https://github.com/swz30/Restormer), [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion), [DM_underwater](https://github.com/piggy2009/DM_underwater). We thank the authors for their outstanding contributions.
