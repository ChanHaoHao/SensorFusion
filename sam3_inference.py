import torch
import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import time
import plotly.graph_objects as go
import numpy as np

# Load the model
start_time = time.time()
model = build_sam3_image_model()
processor = Sam3Processor(model)
print(f"Finished building SAM3 model in {time.time() - start_time:.2f} seconds.")

image_path = '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/'
image_files = sorted(os.listdir(image_path))

text_prompts = ["car", "pedestrian", "cyclist", "traffic light", "train"]
color_prompts = {"car": "red", "pedestrian": "blue", "cyclist": "green", "traffic light": "yellow", "train": "purple"}

for index in range(len(image_files)):
    image_file = image_files[index]
    image = Image.open(os.path.join(image_path, image_file)).convert("RGB")
    
    rgb_fig = go.Figure(go.Image(z=image))

    rgb_fig.update_xaxes(visible=False, constrain='domain')
    rgb_fig.update_yaxes(visible=False, scaleanchor='x')

    rgb_fig.update_layout(
        width=image.size[0],
        height=image.size[1],
        margin=dict(l=0, r=0, t=0, b=0),  # remove all margins
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    inference_state = processor.set_image(image)

    for prompt in text_prompts:
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        print(f"Processed prompt '{prompt}' in {time.time() - start_time:.2f} seconds.")
        start_time = time.time()
        
        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        n_masks = masks.shape[0]
        if n_masks == 0:
            continue
        
        # # image = np.array(image) # H x W x 3
        for instance_id in range(n_masks):
            mask = masks[instance_id].detach().cpu().numpy()  # 1 x H x W, bool
            mask = mask[0]  # H x W
            u, v = np.where(mask > 0)
            rgb_fig.add_trace(
                go.Scatter(
                    x=v,
                    y=u,
                    mode='markers',
                    marker=dict(
                        color=color_prompts[prompt],
                        size=1,
                        opacity=0.5
                    ),
                    showlegend=False
                )
            )
            
        #     np.save(f"masks/0001/img_{index}_{prompt}_{instance_id}.npy", mask.astype(bool))
    rgb_fig.write_image(f"samples/0001/sam3_detection_{index}.png", scale=1)

    print(f"Processed image {index + 1}/{len(image_files)}: {image_file}")