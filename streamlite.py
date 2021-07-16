import streamlit as st
st.title("Face Mask Detection using Machine Learning")
st.header("Face Mask Detection using Machine Learning")
st.text("Upload an Image for image classification as with_mask or without_mask")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
   
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        predictions = predict(image)
        st.write(predictions)
        st.pyplot(fig)  
        
        
def predict(image):
    classifier_model = "https://tfhub.dev/agripredict/disease-classification/1"
    IMAGE_SHAPE = (300, 300,3)
    model = tf.keras.Sequential([
    hub.KerasLayer(classifier_model,input_shape=IMAGE_SHAPE)])
    test_image = image.resize((300,300))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'with_mask',
          'without_mask',
         ]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'with_mask':0,
          'without_mask':0,
          
}
   
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } percent confidence." 
    return result
  
  
if __name__ == "__main__":
    main()
