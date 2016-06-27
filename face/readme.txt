#########################################
#                                       #
#   Face Analysis Library, Jan 2011     #
#                                       #  
#   EApL, Intel Labs China              #
#########################################

========================================================================================
Face library Function
========================================================================================
Computer vision and pattern recognition techniques which aim to detect the size and location of human faces on still image or video streams. Major components include:
  o Face detection: whether or not there is a face in a still image
  o Face tracking:   predict the position of face in motion images for real-time applications
  o Landmark detection: detect the positions of salient points such as eye corner, mouth corner within a face bounding box
  o Face alignment: register a sequence of faces with regard to the landmark points to handle variations of pose, illumination, view points, etc
  o Blink detection: detect the eyes are open or close
  o Smile detection: detect smile expression, which is used in smile shutter of Digital Camera.
  o Gender classification: classify male/female faces.
  o Age classification:  classify baby/kid/adult/senior faces.
  o Face recognition: face verification/identification and retrieval, which can recognize a face's ID/name with a probability score output.

========================================================================================
Face library files description
========================================================================================
(1)Execution required files
  ./bin                              the execution folder, facetest.exe is the running program
  ./bin/opencv_input   face analysis classifier models, sure that it is located in the ./bin folder
  ./bin/face_database  face exemplars database. Each person's face images are in one folder.
  ./bin/video_data        testing video data

(2)Compiling required 3thparty files
  ../3thparty/fann          fast neural network lib for face classfiers
  ../3thparty/fftw           fast fft lib for gabor feature extraction
  ../3thparty/opencv200  openCV APIs for image/video processing
  ../3thparty/pthread_win32  pthread lib for parallel between face tracking and face recognition

(3)Core face lib files in ../face
  [Face detection and tracking]
  #  father class
  o  cxfacedetector.hpp
  o  cxfacetracker.hpp

  #  haar detector and tracker
  o  cxhaarfacedetector.hpp
  o  cxhaarfacedetector.cpp
  o  cxhaarfacetracker.hpp
  o  cxhaarfacetracker.cpp
  o  ./faceutil/cxhaarclassifier.cpp
  o  ./faceutil/cxhaarclassifier.hpp

  #  surf detector and tracker
  o  cxsurffacedetector.hpp
  o  cxsurffacedetector.cpp
  o  cxsurffacetracker.hpp
  o  cxsurffacetracker.cpp
  o  ./feature/integrafea.hpp
  o  ./feature/integrafea.cpp
  o  ./faceutil/khash.hpp
  o  ./faceutil/disjoint-set.hpp
  o  ./faceutil/postfilter.hpp

  [Face 6-pt landmark detection and tracking]
  o  cxcompdetector.hpp
  o  cxcompdetector.cpp

  [Face cascade detection for smile/blink/gender]
  o  smile/blink/gender detector
  o  cxboost.hpp
  o  cxboost.cpp

  [Age detector]
  o  cxmcboost.hpp
  o  cxmcboost.cpp

  [Face recognizer]
  o  cxboostfacerecog.hpp
  o  cxboostfacerecog.cpp
  o  ./faceutil/basetypes.hpp
  o  ./faceutil/facelistxml.hpp
  o  ./faceutil/facelistxml.cpp

  [Facial features]
  o  ./feature/cxslidewinfea.hpp
  o  ./feature/cxslidewinfea.cpp
  o  ./feature/gaborutil.hpp
  o  ./feature/gaborutil.cpp
  o  ./feature/loggabor.hpp
  o  ./feature/loggabor.cpp

  o  ./feature/integrafea.hpp
  o  ./feature/integrafea.cpp

  [Face utilities]
  #  color conversion and align face
  o  ./faceutil/cxfaceutil.hpp
  o  ./faceutil/cxfaceutil.cpp
  
  #  video reader and write
  o  ./faceutil/cxvideocap.hpp
  
  #  argument opt
  o  ./faceutil/cxoptions.hpp


========================================================================================
Contact info:
========================================================================================
Name:  Zhang Yimin; Wang, Tao; Du, Yangzhou; Li, Jianguo; Li, Eric Q
Email: { yimin.zhang, tao.wang, yangzhou.du, jianguo.li, eric.q.li } @ intel.com