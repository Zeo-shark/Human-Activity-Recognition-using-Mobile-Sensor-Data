# Human-Activity-Recognition-using-Mobile-Sensors-Data

Human Actvity Recognition using mobile Sensors data is a research based project to build a model/system using RCNN-LSTM to detect realtime human activity using mobile's motion sensors data like Accelerometer and Gyroscope.
### Our Machine Learning Model
## On Device "Offline" Human Activity Detection Model...               
#### Sensors Data to Image
If the 3-axes of the human motion model are considered as the 3 channels of a RGB image, the value of the XYZ axial data can be mapped into the value of the RGB channel data in a RGB image respectively. Namely, each 3-axial data can be converted into an RGB pixel. The 400 pieces of 3-axial data cached in the sliding window can be viewed as a bitmap with size of 20 or 20 pixels.    
<br/>
![image_2](assets/Capture4.PNG)

#### Requirements
- python3
- tensorflow: >=2.0.0    
- numpy     
- pandas     
- matplotlib

#### Dataset
- 1.MobiFall＆MobiAct DataSet
http://www.bmi.teicrete.gr/index.php/research/mobiact

- 2.SisFall http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/

#### Sensors Data graph
![image_3](/assets/1.png)          
![image_4](/assets/2.png)            
![image_5](/assets/3.png)           
![image_5](/assets/4.png)               

#### Accuracy
Max Training Accuracy: 92.405%             
Max Testing Accuracy: 96.94%



<!-- ### Languages and Tools:

<img align="left" alt="Visual Studio Code" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/visual-studio-code/visual-studio-code.png" />
<img align="left" alt="HTML5" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/html/html.png" />
<img align="left" alt="CSS3" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/css/css.png" />
<img align="left" alt="Sass" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/sass/sass.png" />
<img align="left" alt="JavaScript" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/javascript/javascript.png" />
<img align="left" alt="Node.js" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/nodejs/nodejs.png" />
<img align="left" alt="MongoDB" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/mongodb/mongodb.png" />
<img align="left" alt="Git" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png" />
<img align="left" alt="GitHub" width="26px" src="https://raw.githubusercontent.com/github/explore/78df643247d429f6cc873026c0622819ad797942/topics/github/github.png" />
<img align="left" alt="HTML5" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/terminal/terminal.png" />

<br />



### Connect with us:

<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />
<img align="left" alt="codeSTACKr | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />
<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />


<br /> -->
