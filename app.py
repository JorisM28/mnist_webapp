from PIL import Image
import numpy as np
import gradio as gr

def save_drawing(editor_data):
  if editor_data is None:
    return None

  final_image = editor_data["composite"]  


  img = Image.fromarray(final_image.astype("uint8"))
  img.save("mon_dessin.png")  

  return final_image


with gr.Blocks() as demo:
  gr.Markdown("# MNIST CNN **Predictor**")

  with gr.Row():

    editeur = gr.ImageEditor(
        label="Drawing canvas",
        sources=(),  

        type="numpy",

        brush=gr.Brush(
            colors=["#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00"],
            default_color="#000000",
        ),
        eraser=gr.Eraser(),
    )



    sortie = gr.Image(label="Final Draw exported")


  validate_button = gr.Button("Export the draw", variant="primary")


  validate_button.click(fn=save_drawing, inputs=editeur, outputs=sortie)


demo.launch()