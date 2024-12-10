import gradio as gr
from imageClassifier import predict_image, draw_attention_path
from audioClassifier import classify_audio_file
from ds import combine_beliefs



# Function to classify both image and audio
def classify_multimedia(image, audio):
    if not image or not audio:
        return "Both image and audio are required.", "", "", "", None
    
    # Image classification
    predicted_class_image, confidence_score_image = predict_image(image)

    # Audio classification
    predicted_class_audio, confidence_audio = classify_audio_file(audio)

    # Uncertainty assignment for image and audio classes
    uncertainty_class = "AD" if predicted_class_image == "CN" else "CN"
    uncertainty_class_audio = "AD" if predicted_class_audio == "CN" else "CN"

    # Printing for debugging purposes
    print(uncertainty_class_audio, confidence_audio, predicted_class_image, uncertainty_class)

    # Create belief sets for image and audio predictions
    belief1 = {
        frozenset({predicted_class_audio}): confidence_audio,
        frozenset({uncertainty_class_audio}): 1 - confidence_audio  # Uncertainty
    }

    belief2 = {
        frozenset({predicted_class_image}): confidence_score_image,
        frozenset({uncertainty_class}): 1 - confidence_score_image 
    }
    # Combine beliefs using the provided function
    combined_belief = combine_beliefs(belief1, belief2)
    
    # Output combined belief for debugging
    print("Combined Belief:", combined_belief)
    
    # Generate attention map plot and return path to saved image
    attention_map_path = draw_attention_path(image)

    # Return all outputs, including combined belief and attention map path
    return predicted_class_image, confidence_score_image, predicted_class_audio, confidence_audio, attention_map_path

#Gradio interface 
with gr.Blocks() as demo:
    
    gr.Markdown("# Multimedia Classifier")
    gr.Markdown("### Upload both an image and an audio file to get classification results.")

  
    image_input = gr.Image(type="filepath", label="Upload Image")
    
   
    audio_input = gr.Audio(type="filepath", label="Upload Audio")
    
    
    classify_button = gr.Button("Classify")


    predicted_class_output_image = gr.Textbox(label="Predicted Class (Image)")
    confidence_output_image = gr.Textbox(label="Confidence Score (Image)")
    
    
    predicted_class_output_audio = gr.Textbox(label="Predicted Class (Audio)")
    confidence_output_audio = gr.Textbox(label="Confidence Score (Audio)")
    combined_belief = gr.Textbox(label="Combined belief")

    
    attention_map_output = gr.Image(label="Attention Map")

 
    classify_button.click(
        fn=classify_multimedia,
        inputs=[image_input, audio_input],
        outputs=[
            predicted_class_output_image, 
            confidence_output_image, 
            predicted_class_output_audio, 
            confidence_output_audio, 
            attention_map_output  
        ]
    )

#launch the interface
demo.launch()

