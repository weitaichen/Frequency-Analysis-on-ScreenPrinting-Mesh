# Frequency-Analysis-on-ScreenPrinting-Mesh
This program use Fourier frequency analysis to extract and label mesh on screen printed sensor. This program is using OpenCV3.0 library on Visual studio 2012 platform.


## How can you get started
https://docs.opencv.org/2.4/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html<br />
https://www.youtube.com/watch?v=l4372qtZ4dc<br />
Above are tutorial webisites which tell you how to install OpenCV library on visual studio platform, following are steps:<br />
* Download OpenCV on https://opencv.org/releases.html<br />
* Setting environment variable:PC->Properties->Advanced system setting->Environment variables->Path<br />
->Edit->New->OpenCV bin directory<br />
* Open new projects: Open Visual studio->File->New project->Visual C++->Win32 Console application->Finish<br />
* Choose configuration: Configuration manager->Active solution platform->New->Choose x86 or x64 platform <br />
* Set include path: Project->Properties->C/C++->Additional include directories->Opencv include directory<br />
* Add library directories:Project->Properties->linker->General->Additional library directories->Opencv lib directory<br />
* Add library dependency:Project->Properties->linker->Input->Additional dependecies->opencv_world300d.lib<br />
* Download msvcp120d.dll and msvcr120d.dll to main.exe folder <br />
* Copy all file into your project folder<br />
* Build, run, and have fun! <br />

## How does this work
* The goal of this program is to find and label mesh defects on screen printed sensors. The results of each step are shown below: 

* Input image:
<p align="center"><img src="/image/screenPrinting.png" height="60%" width="60%"></p><br />

* Frequency domain:
input image is transfer to frequnecy domain using discrete Fourier transform function which is built in OpenCV. A manual binary threshold then applied on frequency domain to extract major response in frequency domain. 
<p align="center"><img src="/image/fequencyDomain.jpg" height="60%" width="60%"></p><br />

* Extract mesh area
The reconstructed mesh area is shifting in horizontal direction. Correct mesh position is extracted with maximum mean gray-level of input image in mesh area.
<p align="center"><img src="/image/detectedMesh.png" height="45%" width="45%"></p><br />

* Cropped image
Since there is losing information near edge on the reconstructed mesh area. So a half-size image is cropped from center of input image.
<p align="center"><img src="/image/croppedImage.png" height="45%" width="45%"></p><br />

* Labeled image
The result is the detected and labeled mesh detects area. The psudo colors are used to visualized separate area in different labels.
<p align="center"><img src="/image/labeledImage.png" height="45%" width="45%"></p>
