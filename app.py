import os
import sys
import os.path as osp
from pathlib import Path
import cv2
import gradio as gr
import torch
import math

try:
    import mmpose
except:
    os.system('pip install /home/user/app/main/transformer_utils')
# os.system('pip install /home/user/app/main/transformer_utils')

# DEFAULT_MODEL='smpler_x_h32'
# OUT_FOLDER = '/home/user/app/demo_out'

# os.system('pip install main/transformer_utils')

DEFAULT_MODEL='smpler_x_h32'
OUT_FOLDER = 'demo_out'
os.makedirs(OUT_FOLDER, exist_ok=True)
num_gpus = 1 if torch.cuda.is_available() else -1
from main.inference import Inferer
inferer = Inferer(DEFAULT_MODEL, num_gpus, OUT_FOLDER)

def infer(video_input, in_threshold=0.5):
    os.system(f'rm -rf {OUT_FOLDER}/*')
    all_mesh_paths = []
    all_smplx_paths = []
    cap = cv2.VideoCapture(video_input)
    fps = math.ceil(cap.get(5))
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_path = osp.join(OUT_FOLDER, f'out.mp4')
    video_output = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    success = 1
    frame = 0
    while success:
        success, original_img = cap.read()
        if not success:
            break
        frame += 1
        img, mesh_paths, smplx_paths = inferer.infer(original_img, in_threshold, frame)
        video_output.write(img)
        all_mesh_paths.extend(mesh_paths)
        all_smplx_paths.extend(smplx_paths)
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
    return video_path, all_mesh_paths, all_smplx_paths

TITLE = '''<h1 align="center">SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation</h1>'''
VIDEO = '''
<center><iframe width="960" height="540" 
src="https://www.youtube.com/embed/DepTqbPpVzY?si=qSeQuX-bgm_rON7E"title="SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen>
</iframe>
</center><br>'''
DESCRIPTION = '''
<b>Official Gradio demo</b> for <a href="https://caizhongang.com/projects/SMPLer-X/"><b>SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation</b></a>.<br>
<li><a href="https://arxiv.org/pdf/2309.17448.pdf">[NeurIPS 2023 (Dataset and Benchmark Track)]</a></li>
<li><a href="https://github.com/caizhongang/SMPLer-X">[code.git]</a></li><br>
<p>
Note: You can drop a video at the panel (or select one of the examples) 
    then you will get the 3D reconstructions of the detected human. ).
</p>
'''

with gr.Blocks(title="SMPLer-X", css=".gradio-container") as demo:

    gr.Markdown(TITLE)
    gr.HTML(VIDEO)
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input video", elem_classes="video")
        with gr.Column():
            threshold = gr.Slider(0, 1.0, value=0.5, label='Bbox Detection Threshold')
            send_button = gr.Button("Infer")
    gr.HTML("""<br/>""")

    with gr.Row():
        with gr.Column():
            video_output = gr.Video(elem_classes="video")
        with gr.Column():
            meshes_output = gr.File(label="3D meshes")
            smplx_output = gr.File(label= "smplx models")
    # example_images = gr.Examples([])
    send_button.click(fn=infer, inputs=[video_input, threshold], outputs=[video_output, meshes_output, smplx_output])
    # with gr.Row():
    # example_images = gr.Examples([
    #     ['/home/user/app/assets/test1.jpg'], 
    #     ['/home/user/app/assets/test2.jpg'], 
    #     ], 
    #     inputs=[input_image, 0.5])

#demo.queue()
demo.launch(debug=True, share=True)
