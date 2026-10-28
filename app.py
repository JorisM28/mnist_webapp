import gradio as gr
from PIL import Image


from preprocess import preprocess, preprocessing
from predictor import DigitPredictor

predictor = DigitPredictor("weights/simple_cnn.pt")

def process_and_predict(editor_data):
    if editor_data is None:
        return None, {}

    final_image = editor_data["composite"]  
    img_pil = Image.fromarray(final_image.astype("uint8"))

    img_clean = preprocess(img_pil)

    tensor = preprocessing(img_clean)

    predictions_dict = predictor.predict(tensor)

    return img_clean, predictions_dict



with gr.Blocks() as demo:
    gr.Markdown("# MNIST CNN **Predictor**")

    with gr.Row():

        editeur = gr.ImageEditor(
            label="Zone de dessin",
            sources=(),  
            type="numpy",
            brush=gr.Brush(colors=["#000000"], default_color="#000000"),
            eraser=gr.Eraser(),
        )


        with gr.Column():
            vue_modele = gr.Image(
                label="Ce que voit le modèle (28x28)", 
                image_mode="L", 
                width=200, 
                height=200
            )
            
            resultat_label = gr.Label(label="Top Probabilités")

    btn_predire = gr.Button("Analyser le chiffre", variant="primary")

    btn_predire.click(
        fn=process_and_predict, 
        inputs=editeur, 
        outputs=[vue_modele, resultat_label]
    )

demo.launch()