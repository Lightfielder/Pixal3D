import os
import argparse
import torch
import gc
import numpy as np
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.7"
os.environ["ATTN_BACKEND"] = "flash_attn_3"

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")

# ============================================================================
# Main Inference
# ============================================================================

def run_inference_memory_efficient(
    image_path: str,
    output_path: str,
    seed: int = 42,
    max_num_tokens: int = 4096,
):
    clean_memory()
    
    # Load pipeline
    print(f"[Pipeline] Loading from TencentARC/Pixal3D...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained("TencentARC/Pixal3D")
    
    if pipeline is None:
        raise RuntimeError("Failed to load pipeline")
    
    print(f"[Pipeline] Loaded successfully")
    
    # Set low VRAM mode
    pipeline.low_vram = True
    print(f"[Pipeline] low_vram = {pipeline.low_vram}")
    
    # Move to GPU
    print("[Pipeline] Moving to GPU...")
    pipeline.to('cuda')
    print("[Pipeline] On GPU")
    
    clean_memory()
    
    # Import the feature extractor
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    
    # Load SS model
    print("[ImageCond] Loading SS model...")
    ss_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    }
    pipeline.image_cond_model_ss = DinoV3ProjFeatureExtractor(**ss_config)
    pipeline.image_cond_model_ss.to('cuda')
    pipeline.image_cond_model_ss.eval()
    print("  SS model loaded")
    clean_memory()
    
    # Preprocess image
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    image_preprocessed = pipeline.preprocess_image(img)
    
    # Default camera parameters (skip MoGe to save memory)
    camera_params = {
        'camera_angle_x': 0.8575,
        'distance': 1.8,
        'mesh_scale': 1.0
    }
    print(f"  Using default camera parameters")
    
    # Load shape model 512 (required for cascade)
    print("[ImageCond] Loading Shape 512 model...")
    shape_512_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    }
    pipeline.image_cond_model_shape_512 = DinoV3ProjFeatureExtractor(**shape_512_config)
    pipeline.image_cond_model_shape_512.to('cuda')
    pipeline.image_cond_model_shape_512.eval()
    
    if hasattr(pipeline.image_cond_model_shape_512, 'use_naf_upsample') and pipeline.image_cond_model_shape_512.use_naf_upsample:
        print("  Loading NAF upsampler for shape 512...")
        pipeline.image_cond_model_shape_512._load_naf()
    clean_memory()
    
    # Load shape model 1024
    print("[ImageCond] Loading Shape 1024 model...")
    shape_1024_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    }
    pipeline.image_cond_model_shape_1024 = DinoV3ProjFeatureExtractor(**shape_1024_config)
    pipeline.image_cond_model_shape_1024.to('cuda')
    pipeline.image_cond_model_shape_1024.eval()
    
    if hasattr(pipeline.image_cond_model_shape_1024, 'use_naf_upsample') and pipeline.image_cond_model_shape_1024.use_naf_upsample:
        print("  Loading NAF upsampler for shape 1024...")
        pipeline.image_cond_model_shape_1024._load_naf()
    clean_memory()
    
    # Load texture model
    print("[ImageCond] Loading Texture model...")
    tex_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    }
    pipeline.image_cond_model_tex_1024 = DinoV3ProjFeatureExtractor(**tex_config)
    pipeline.image_cond_model_tex_1024.to('cuda')
    pipeline.image_cond_model_tex_1024.eval()
    
    if hasattr(pipeline.image_cond_model_tex_1024, 'use_naf_upsample') and pipeline.image_cond_model_tex_1024.use_naf_upsample:
        print("  Loading NAF upsampler for texture...")
        pipeline.image_cond_model_tex_1024._load_naf()
    clean_memory()
    
    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)
    
    ss_sampler_override = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    shape_sampler_override = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    tex_sampler_override = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}
    
    try:
        mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
            image_preprocessed,
            camera_params=camera_params,
            seed=seed,
            sparse_structure_sampler_params=ss_sampler_override,
            shape_slat_sampler_params=shape_sampler_override,
            tex_slat_sampler_params=tex_sampler_override,
            preprocess_image=False,
            return_latent=True,
            pipeline_type="1024_cascade",
            max_num_tokens=max_num_tokens,
        )
        
        # Unload models after pipeline.run completes
        print("[Memory] Unloading models...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512', 
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            if hasattr(pipeline, attr) and getattr(pipeline, attr) is not None:
                model = getattr(pipeline, attr)
                model.to('cpu')
                del model
                setattr(pipeline, attr, None)
        clean_memory()
        
        mesh = mesh_list[0]
        
        # Extract GLB with smaller texture size
        print("[Inference] Extracting GLB...")
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=200000, texture_size=1024,
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
        )
        
        # Apply rotation
        rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
        glb.apply_transform(rot)
        
        # Export
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        glb.export(output_path, extension_webp=True)
        print(f"[Done] GLB saved to: {output_path}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[ERROR] Out of memory. Try reducing max_num_tokens further:")
            print(f"  python inference_final.py --image {image_path} --max_num_tokens 2048")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Memory-Efficient Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_num_tokens", type=int, default=4096, help="Max tokens (lower = less memory)")
    
    args = parser.parse_args()
    
    run_inference_memory_efficient(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        max_num_tokens=args.max_num_tokens,
    )
