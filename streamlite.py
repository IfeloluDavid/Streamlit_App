
EXTERNAL_DEPENDENCIES = {
"export_5.pkl": {
"url": "https://www.dropbox.com/s/uhf8ioorhkqz1gy/mymodel?dl=0",
"size": 246698366}
}

for filename in EXTERNAL_DEPENDENCIES.keys():
    download_file(filename)


# FUNCTION TO UPLOAD A FILE
uploaded_file = st.file_uploader("upload a black&white photo", type=['jpg','png','jpeg'])

if uploaded_file is not None:
   g = io.BytesIO(uploaded_file.read())  # BytesIO Object
   temporary_location = "image/temp.jpg"
   
   with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
      out.write(g.read())  # Read bytes into file
   
      # close file
      out.close()
        
        
        
        
# FUNCTION TO RESIZE THE IMAGE        
def resize_one(fn,img_size=800):
   dest = 'image/image_bw/temp_bw.jpg'
   
   # Load the image
   img=cv2.imread(str(fn))
   height,width  = img.shape[0],img.shape[1]
if max(width, height)>img_size:
   if height > width:
      width=width*(img_size/height)
      height=img_size
      img=cv2.resize(img,(int(width), int(height)))
   elif height <= width:
      height=height*(img_size/width)
      width=img_size
      img=cv2.resize(img,(int(width), int(height)))
cv2.imwrite(str(dest),img)



# LOAD THE MODEL
def create_learner(path,file):
   learn_gen=load_learner(path,file)
   return learn_gen




def predict_img(fn,learn_gen,img_width=640):
   _,img,b=learn_gen.predict(open_image(fn))
   img_np=image2np(img)
   st.image(img_np,clamp=True,width=img_width)



def main():
   for filename in EXTERNAL_DEPENDENCIES.keys():
      download_file(filename)
   st.title("Black&White Photos Colorisation")

   uploaded_file = st.file_uploader("upload a black&white photo", type=['jpg','png','jpeg'])
   
   if uploaded_file is not None:
       g = io.BytesIO(uploaded_file.read())  # BytesIO Object
       temporary_location = "image/temp.jpg"
   
       with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
          out.write(g.read())  # Read bytes into file
          # close file
          out.close()
    resize_one("image/temp.jpg",img_size=800)
    st.image("image/temp.jpg",width=800)
    
    
    
    start_analyse_file = st.button('Analyse uploaded file')
    if start_analyse_file== True:
        learn_gen=create_learner(path='',file='export_5.pkl')
        predict_img("image/image_bw/temp_bw.jpg",learn_gen,img_width=800)
    
    
    
    
    
if __name__ == "__main__":
    main()
