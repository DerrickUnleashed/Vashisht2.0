from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

def classify_audio(audio_file_path):
    """
    Classify an audio file using the Whisper fine-tuned model
    
    Parameters:
    audio_file_path (str): Path to the audio file to classify
    
    Returns:
    dict: Classification results with probabilities
    """
    # Load model and processor
    processor = AutoProcessor.from_pretrained("b-brave/whisper-small-ft-balbus-sep28k-v1.5")
    model = AutoModelForAudioClassification.from_pretrained("b-brave/whisper-small-ft-balbus-sep28k-v1.5")
    
    # Load and preprocess audio
    print(f"Loading audio file: {audio_file_path}")
    try:
        # Use librosa to load audio file
        audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
        
        # Convert to correct format
        audio_array = audio_array.astype(np.float32)
        
        # Process audio with the processor
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get logits from the output
        logits = outputs.logits
        
        # Convert to probabilities with softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predicted class ID
        predicted_class_id = torch.argmax(probs, dim=-1).item()
        
        # Get the class labels if available
        if hasattr(model.config, "id2label"):
            id2label = model.config.id2label
            predicted_class_label = id2label[predicted_class_id]
        else:
            predicted_class_label = f"CLASS_{predicted_class_id}"
        
        # Create result dictionary
        result = {
            "predicted_class": predicted_class_label,
            "predicted_class_id": predicted_class_id,
            "confidence": probs[0][predicted_class_id].item(),
            "all_probabilities": {
                id2label[i] if hasattr(model.config, "id2label") else f"CLASS_{i}": prob.item()
                for i, prob in enumerate(probs[0])
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return {"error": str(e)}

def visualize_results(result):
    """
    Visualize classification results with a bar chart
    
    Parameters:
    result (dict): Classification results from classify_audio function
    """
    if "error" in result:
        print(f"Cannot visualize results due to error: {result['error']}")
        return
    
    # Extract data for visualization
    labels = list(result["all_probabilities"].keys())
    probs = list(result["all_probabilities"].values())
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]
    
    # Plot top 5 or all if less than 5
    top_n = min(5, len(sorted_labels))
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), sorted_probs[:top_n], color='skyblue')
    plt.xticks(range(top_n), sorted_labels[:top_n], rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Top Classification Results')
    plt.tight_layout()
    plt.show()
    
    # Print prediction details
    print(f"\nPredicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")

# Example usage
if __name__ == "__main__":
    # Ask user for audio file input
    audio_path = input("Enter the path to your audio file: ")
    
    # Check if file exists
    if not Path(audio_path).exists():
        print(f"Error: File '{audio_path}' does not exist.")
    else:
        # Classify audio
        classification_result = classify_audio(audio_path)
        
        # Display results
        if "error" not in classification_result:
            print("\nClassification Results:")
            print(f"Predicted class: {classification_result['predicted_class']}")
            print(f"Confidence: {classification_result['confidence'] * 100:.2f}%")
            
            # Ask if user wants to see visualization
            if input("\nShow visualization? (y/n): ").lower() == 'y':
                visualize_results(classification_result)
        else:
            print(f"Classification failed: {classification_result['error']}")