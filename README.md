1. Alignment: python alignment.py --src "path of testing images" --des "saved aligned images"  
Please make sure your testing images look like this RGB_wild/subject1/img1,img2,...
　　　　　　　　　　　　　　　　　　　　　　　　　　 /subject2/img1,img2,...  
The aligned images will look like this RGB/subject1/img1,img2,...    
　　　　　　　　　　　　　　　　　　/subject2/img1,img2,...

2. Put two downloaded ckpts under ckpt/

3. Run python demo.py --rgb "your aligned rgb input path" --type "jpg or png"  
Please make sure your aligned rgb input path look like this "*****/RGB/"

4. The output depth images will be in "*****/D_out/"
